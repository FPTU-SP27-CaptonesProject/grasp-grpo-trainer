# Quy trình train GRPO có cổng kiểm tra và early stopping

- **Ngày:** 2026-11-04
- **Repo:** `grasp-grpo-trainer`
- **Bối cảnh:** run `grpo-gold-evidence` chạy 3.089 bước (~70 giờ GPU) nhưng reward, F1 và recall đi ngang,
  còn KL tăng dần (0,02 → 0,06). Ước tính còn ~77 giờ nữa mới xong. Spec này sửa nguyên nhân, và đặt
  quy trình để mỗi giờ GPU chỉ được chi khi chặng trước đã chứng minh có tín hiệu.

## 1. Chẩn đoán run hiện tại

| # | Vấn đề | Vị trí | Hệ quả |
|---|---|---|---|
| 1 | `LEARNING_RATE=5e-7` cho LoRA. 5e-7 là lr full finetune của SPARKLE; SFT LoRA dùng 2e-4 | `config/vast.env.example:26`, mặc định `train.py` | Với AdamW, mỗi bước dịch mỗi tham số ~lr. Sau 3.089 bước dịch tối đa ~0,0015, so với ~0,32 của SFT (gấp ~200 lần). Adapter gần như đứng yên |
| 2 | `QUESTIONS_PER_ITERATION=1`, `MAX_LEAVES=4` | `config/vast.env.example:21` | Mỗi cập nhật ~12 action, SPARKLE dùng batch 1024. Gradient nhiễu hơn ~9 lần (1/√n), Adam đi kiểu bước ngẫu nhiên: KL tăng, reward không tăng |
| 3 | `ALPHA=0.7` cộng reward shaping của mọi action trên đường đi | `environment.py:47`, `environment.py:206` | 70% reward là context recall, F1 chỉ 30%; shaping đẩy reward lên ~3,5 và lấn át F1 |
| 4 | Prompt reader tự viết, khác `grasp/src/grasp/agents/prompts/reader.txt` dùng khi sinh dữ liệu SFT; định dạng context khác (`[doc_id] title` so với `Title:`/`Text:`) | `environment.py:17` | Proxy nhận thought khác phân phối SFT; nghi là lý do F1 chỉ ~0,2 |
| 5 | `MAX_STEPS=3`, lúc sinh SFT là 5 | `config/vast.env.example:24`, `grasp/configs/collect/sft.yaml` | Câu multi-hop bị cắt trước khi đủ bằng chứng |
| 6 | Ratio và KL tính trên tổng log-prob cả câu; KL bị chặn `min=0` | `grpo.py:44`, `grpo.py:46` | Ratio cực đoan với QF dài khi lr đủ lớn; KL chỉ phạt một chiều |
| 7 | Actor và reference nạp 4-bit; sinh 1 prompt một lần | `scripts/train_grpo_vast.sh`, `model.py:70` | ~50 giây/câu hỏi; GPU rảnh phần lớn thời gian |

Giả thuyết 1–3 giải thích việc không học; 4–5 giải thích F1 thấp. Chặng 0 và 2 dưới đây kiểm chứng từng giả thuyết
trước khi tốn giờ GPU cho run dài.

## 2. Phạm vi

**Làm:**
- Tăng tốc rollout (bf16, gom batch sinh văn bản) mà không đổi thuật toán.
- Đưa môi trường về khớp lúc sinh SFT (prompt reader, định dạng context, `max_steps`).
- Sửa objective GRPO: loss theo token, KL k3, advantage không chia std, lọc nhóm không có tín hiệu, clip-higher, warmup.
- Eval định kỳ và early stopping tự động.
- Lệnh và cổng cho 4 chặng (mục 5).

**Không làm (spec sau):**
- Rẽ nhánh ở QF/KI hoặc chia điểm thưởng theo agent. Cây vẫn chỉ rẽ ở RD.
- Chuyển sang TRL/verl hoặc vLLM cho proxy.
- Sửa `grasp-gold-evidence-builder` (6.374/9.000 câu có `gold_doc_ids`). Câu thiếu gold vẫn dùng reward chỉ F1.

## 3. Thay đổi

### 3.1 Tăng tốc

- **bf16 mặc định.** Bỏ `--load-in-4bit` khỏi `scripts/train_grpo_vast.sh`. Actor và reference 3B bf16 tốn ~12GB,
  vừa card 24GB. Adapter SFT train trên base 4-bit vẫn nạp được lên base bf16; chặng 0 đo F1 của SFT trên bf16 để
  xác nhận không tụt.
- **Gom batch sinh văn bản qua nhiều cây.** Thêm `GenerationBatcher` (file mới `batcher.py`):
  - Mỗi cây chạy trong một thread; reader (vLLM) vốn nhận request song song.
  - Thread gọi `batcher.generate(role, prompt, samples)` thay vì `model.generate`. Batcher gom request cùng role
    (cùng `max_new_tokens`) trong cửa sổ ~20ms hoặc khi đủ `max_batch`, sinh một lần, trả kết quả về từng thread.
  - Kết quả phải giống hệt bản tuần tự khi `temperature=0` (test ở mục 7).
- **Mục tiêu:** ≤ 15 giây/câu hỏi tính bình quân khi chạy 16 cây song song. Chặng 0 đo con số thật; nếu > 20 giây thì
  tính lại chi phí pilot trước khi đi tiếp.

### 3.2 Môi trường khớp lúc sinh SFT

- `reader_prompt` dựng từ đúng nội dung `reader.txt` của `grasp` (copy văn bản vào `contracts.py`, có test so khớp
  từng byte như prompt RD/QF/KI). Định dạng context `Title: …\nText: …`.
- `MAX_STEPS=5`.
- Reader của Project A không gọi chat mà gọi text completion: prompt = chat template(reader.txt với context hiện tại)
  + transcript các thought trước, dừng ở `</think>`; hết bước thì sinh lại câu trả lời từ đầu với context cuối.
  Trainer làm đúng giao thức này.
- Cờ `--reader-protocol {grasp,legacy}` giữ prompt cũ để chặng 0 so sánh; mặc định `grasp`.

### 3.3 Objective GRPO

| Thành phần | Hiện tại | Mới |
|---|---|---|
| Ratio | `exp(Σ logp − Σ logp_old)` cả câu | `exp(logp_t − logp_old_t)` từng token (đã có `old_token_logprobs` trong rollout) |
| Clip | ε = 0,2 hai phía | ε_low = 0,2, ε_high = 0,28 (clip-higher) |
| Gộp loss | trung bình theo action | tổng theo token của cả batch / tổng số token (token-level) |
| KL | `clamp(logp − logp_ref, min=0)` trên cả câu | k3 theo token: `exp(r) − r − 1`, `r = logp_ref_t − logp_t`; β = 0,002 |
| Advantage | `(R − mean) / std` trong cây | `R − mean` (cờ `--advantage-std-norm` để bật lại) |
| Nhóm reward giống hệt nhau | advantage = 0, vẫn tính vào bước | bỏ khỏi batch, lấy câu khác bù tới khi đủ `questions_per_iteration` hoặc hết `max_refill` (mặc định 2×) |
| Reward | α = 0,7, shaping đầy đủ | α = 0,3; shaping nhân `--shaping-scale` = 0,1 |
| Lịch lr | hằng số | warmup tuyến tính 20 bước, sau đó hằng số |
| Optimizer | AdamW | giữ AdamW, grad clip 1,0 |

### 3.4 Siêu tham số mặc định cho pilot

| Tham số | Giá trị |
|---|---|
| `LEARNING_RATE` | 1e-5 |
| `QUESTIONS_PER_ITERATION` | 16 |
| `MAX_LEAVES` (G) | 4 |
| `MAX_STEPS` | 5 |
| `TEMPERATURE` / `TOP_P` | 0,7 / 0,95 (giữ nguyên) |
| `ALPHA` | 0,3 |
| `SHAPING_SCALE` | 0,1 |
| `KL_BETA` | 0,002 |
| `EVAL_EVERY` | 50 bước |
| `EVAL_QUESTIONS` | 300 câu val cố định |

## 4. Eval định kỳ

- **Tập eval:** 300 câu cố định, lấy ngẫu nhiên (seed 42) từ `grasp-gold-evidence/dev.jsonl`, 100 câu mỗi dataset,
  chỉ lấy câu có `gold_doc_ids`; ghi id vào `config/eval_ids.json` và commit. Không trùng câu train.
- **Cách chạy:** greedy (`temperature=0`), 1 đường, không rẽ nhánh, cùng môi trường như train.
- **Số đo:** EM, F1, context recall, tỷ lệ output đúng format theo vai, số lần retrieve trung bình.
- **Độ nhiễu:** F1 từng câu có độ lệch chuẩn ~0,4; với 300 câu so sánh cặp, sai số chuẩn ~0,01–0,015. Vì vậy
  ngưỡng "có cải thiện" là **+0,02 F1**.
- Kết quả ghi vào `eval_history.json`; checkpoint có F1 eval cao nhất được copy sang `best/`.

## 5. Bốn chặng

| Chặng | Lệnh | Thời gian ước | Cổng đi tiếp | Không đạt thì |
|---|---|---|---|---|
| **0. Mốc SFT** | `grasp-eval-policy` với adapter SFT, 300 câu eval, chạy 2 lần (`--reader-protocol grasp` và `legacy`). Đo thêm giây/câu và tỷ lệ nhóm reward giống hệt nhau khi sample G = 4 | 1–2 giờ | F1 bản `grasp` ≥ 0,30 (cao hơn rõ mức ~0,2 của run cũ), nhóm cùng reward ≤ 50%, giây/câu ≤ 20 | Sửa môi trường hoặc tốc độ; chưa train |
| **1. Smoke** | 20 bước, 4 câu/bước | ~30 phút | Không NaN; ratio trung bình trong [0,9; 1,1]; KL > 0 và < 0,05; grad norm > 0; trọng số LoRA thay đổi (so `adapter_model.safetensors` trước/sau) | Sửa lỗi code |
| **2. Overfit** | 40 bước, lặp trên 8 câu cố định, 8 câu/bước | ~1 giờ | Reward trung bình 10 bước cuối thu hẹp ≥ 30% khoảng cách từ reward 10 bước đầu tới 1,0 trên chính 8 câu đó (sửa 2026-11-08: SFT đã đạt 0,84 nên "+30%" vượt trần 1,0) | Lỗi ở advantage, loss hoặc lr: dừng, debug |
| **3. Pilot** | 300 bước, cấu hình mục 3.4 | 15–20 giờ | Có ít nhất một lần eval F1 ≥ mốc SFT + 0,02, và không luật dừng nào kích hoạt | Giữ báo cáo, quyết định lại cấu hình |
| **4. Full** | Chạy tiếp từ `best/` của pilot (`--actor-checkpoint`), reference vẫn là SFT | Theo `MAX_HOURS` | Early stopping quyết định | — |

Lệnh từng chặng nằm trong `scripts/stage{0..3}_*.sh`, mỗi script đọc `config/vast.env` và ghi
`outputs/<run>/stage_report.json` có trường `passed` và lý do.

## 6. Early stopping

Kiểm tra sau mỗi bước (luật dựa trên train) và sau mỗi lần eval (luật dựa trên eval). Luật đầu tiên kích hoạt
sẽ dừng run.

| Luật | Điều kiện | Hành động |
|---|---|---|
| Không học | 3 lần eval liên tiếp (150 bước) F1 không vượt mức tốt nhất ≥ 0,02 | Dừng, giữ `best/` |
| Tệ hơn SFT | F1 eval thấp hơn mốc SFT ≥ 0,03 trong 2 lần eval liên tiếp | Dừng |
| Sụp format | Tỷ lệ đúng format < 90%, trung bình 20 bước gần nhất | Dừng |
| Trôi quá xa | KL k3 trung bình mỗi token > 0,3, hoặc entropy < 30% mức 20 bước đầu | Dừng |
| Hết tín hiệu | > 60% nhóm có reward giống hệt nhau, trung bình 20 bước gần nhất | Dừng |
| Trần ngân sách | Thời gian chạy > `MAX_HOURS` | Lưu checkpoint, dừng |

- Mọi ngưỡng là tham số CLI, giá trị trên là mặc định.
- Luật "không học" và "tệ hơn SFT" chỉ áp dụng sau warmup và sau lần eval đầu tiên.
- Khi dừng: ghi `stop_reason`, bước, số đo lúc dừng vào `run_manifest.json`; checkpoint dùng cho chặng sau là
  `best/`, không phải checkpoint cuối.

## 7. Log và kiểm thử

**Log mỗi bước** (thêm vào `metrics.json`): `kl_token_k3`, `entropy`, `ratio_mean`, `clip_frac`, `grad_norm`,
`lr`, `format_rate` theo vai, `zero_signal_groups`, `refilled_questions`, `tokens`, `seconds_per_question`.

**Test mới (pytest, chạy trên CPU với model tí hon hoặc giả):**
- `GenerationBatcher` trả kết quả giống bản tuần tự khi `temperature=0`.
- Loss token-level: với ratio = 1 và advantage A, gradient bằng −A·∇logp trung bình theo token.
- KL k3 = 0 khi actor trùng reference, > 0 khi khác.
- Nhóm reward giống hệt nhau bị loại và câu bù được lấy.
- Prompt reader khớp từng byte `reader.txt` của `grasp`.
- Mỗi luật early stopping kích hoạt đúng trên chuỗi số đo giả.

## 8. Chi phí dự kiến

| Chặng | Giờ GPU |
|---|---|
| 0–2 | ~3 |
| 3 (pilot) | 15–20 |
| **Tổng tới lúc biết GRPO có học không** | **~20** |

So với ~150 giờ của run hiện tại. Con số 15–20 giờ dựa trên mục tiêu ≤ 15 giây/câu ở mục 3.1; chặng 0 đo con số
thật và tính lại trước khi chạy pilot.

## 9. Rủi ro

- **Tăng tốc không đạt mục tiêu:** pilot đắt hơn. Phương án: giảm `QUESTIONS_PER_ITERATION` xuống 8 và tăng
  `EVAL_EVERY`, hoặc chuyển proxy sang vLLM (spec riêng).
- **Mốc SFT thấp ngay cả với prompt `grasp`:** lỗi nằm ở bản SFT hoặc pipeline eval, không ở GRPO. Dừng ở chặng 0.
- **lr 1e-5 vẫn quá nhỏ hoặc quá lớn:** chặng 2 phát hiện. Thử 5e-6 và 2e-5 trên chặng 2 (~1 giờ mỗi lần) trước pilot.
- **Bỏ chia std làm advantage lệch thang giữa câu có và không có gold:** theo dõi; bật lại `--advantage-std-norm` nếu
  câu thiếu gold chiếm phần lớn gradient.
