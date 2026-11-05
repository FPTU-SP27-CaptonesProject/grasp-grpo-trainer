# GRASP GRPO Trainer

Standalone Project B for training the Project A Qwen2.5-3B SFT adapter with binary-tree GRPO.
It reproduces the Project A RD/QF/KI prompts, parser contracts, E5+FAISS passage retrieval,
SQLite triple cache use, frozen reader protocol and reward shape. It does **not** rerun SFT.

## What it produces

`grasp-train-grpo` writes a standard PEFT adapter checkpoint:

```text
outputs/grpo/checkpoint-grpo-step-000100/
  adapter_model.safetensors
  adapter_config.json
  tokenizer_config.json
  grpo_config.json
```

That directory is the final proxy policy for GRASP. Keep the matching retrieval artifacts,
reader model and prompt code alongside it for inference and fair baseline evaluation.

## Cloud setup

For Vast.ai, copy the supplied config and run the wrappers rather than manually writing long
commands:

```bash
cp config/vast.env.example config/vast.env
# Edit config/vast.env, then:
bash scripts/setup_vast.sh
bash scripts/validate_vast.sh
# In a second tmux pane: bash scripts/run_reader_vast.sh
bash scripts/smoke_tree_vast.sh
bash scripts/train_grpo_vast.sh
```

`READER_GPU=0` and `TRAIN_GPU=1` target a single two-GPU Vast instance. The 3B proxy generates on a
second vLLM server on GPU 0 (`bash scripts/run_policy_vast.sh` in another tmux window, started after
the reader is up; `POLICY_URL`). It serves the SFT adapter as `sft`; after every optimizer step the
trainer saves the adapter under `<output>/live/` and the server loads it at runtime, so rollouts are
always on-policy. The loss uses its own detached log-probs as the old ones (one update per rollout
batch), and `sampler_logprob_gap` in the step log tracks HF/vLLM agreement; stage 1 fails if it is
large. Leave `POLICY_URL` empty to fall back to HF `generate` on the training GPU. Set `READER_URL` to the
reader instance's mapped public URL only when the reader is hosted on a separate Vast instance.
The setup script creates a separate `READER_VENV` for vLLM because its dependency versions conflict
with the stable Transformers/PEFT stack used by the GRPO trainer.

Download the adapter, corpora, triple cache and gold-evidence QA with
`bash scripts/fetch_data_vast.sh`, then use `CORPUS_CONFIG=config/corpora.vast.json`. Retrieval
reproduces Project A (`configs/retriever/e5_large.yaml` there): FAISS with `nprobe=64` returns 100
candidates, which are re-scored by re-encoding `title. text`, keeping the top 10. Triples come from
Project A's shared cache (`doancuavahk/grasp-retrieval`, `cache/triples.sqlite`). About 21% of
retrieved documents are missing from it; with `EXTRACT_TRIPLES=1` (`--extract-triples`) they are
extracted on first retrieval with Project A's extractor (same prompt, Qwen3-235B on Bedrock,
`BEDROCK_API_KEY`) and written back to the cache, capped per process by `EXTRACT_BUDGET_USD`.
`grasp-eval-policy` reports `triple_coverage` (`cache_misses` = before extraction) and
`triple_extraction` (documents, tokens, USD). Upload the grown cache back to the Hub before the
instance is destroyed so later runs start from it. `scripts/build_triple_caches_vast.sh` (triples from teacher
trajectories only) is kept for the legacy layout.

## Live dashboard

Every stage script pushes its small logs (per-step `metrics.jsonl`, `eval_history.json`, stage
reports, GPU memory samples, the last console lines; never checkpoints) to the private dataset
`LOG_REPO` (default `doancuavahk/grasp-grpo-logs`) every `LOG_SYNC_SECONDS`, plus once on exit.
On your own machine, after `hf auth login`:

```powershell
.venv\Scripts\python scripts\dashboard.py   # opens http://127.0.0.1:8765, refreshes every minute
```

Logs stay on the Hub when the Vast instance stops. Set `LOG_SYNC=0` in `config/vast.env` to disable.

## Gold-evidence GRPO data

For recall-aware GRPO, enrich the QA rows with `gold_doc_ids`. These IDs are used only by the
reward after rollout, never inserted into an actor or reader prompt. `grasp-build-gold-qa` accepts
one SPARKLE-style benchmark annotation file per corpus. An annotation record must contain a
question, answers, and either `gold_ctxs` (with an `id` or `doc_id`) or supporting-fact titles.
It verifies every resolved ID against the target corpus `passages.sqlite`, excludes unresolved or
ambiguous rows, and writes an audit report.

```bash
grasp-build-gold-qa \
  --qa /workspace/data/grpo_train_mixed_shuffled.jsonl \
  --corpus-config /workspace/configs/corpora.json \
  --gold-source 2wiki_ircot=/workspace/data/grasp-gold/raw/2wiki/train.json \
  --gold-source hotpotqa_abstracts_2017=/workspace/data/grasp-gold/raw/hotpotqa/train.json \
  --gold-source wiki18=/workspace/data/grasp-gold/raw/nq/train.json \
  --output /workspace/data/grpo_train_with_gold.jsonl \
  --report /workspace/data/grpo_train_with_gold.report.json \
  --split train
```

Run the same command with a held-out QA file and source dev annotations to create
`grpo_dev_with_gold.jsonl`. Do not use unresolved rows as F1-only training examples. Review the
report and require `rows_written`/coverage to be acceptable and every mapped ID to be valid before
starting GRPO. With gold IDs available, set `ALPHA=0.7` to use the SPARKLE-like mixture
`0.3 * answer_F1 + 0.7 * context_recall`.

When the dev annotation itself contains questions, answers and `gold_ctxs`, create a canonical
evaluation file directly rather than first converting it by hand:

```bash
grasp-build-gold-qa --from-gold \
  --corpus-config /workspace/configs/corpora.json \
  --gold-source 2wiki_ircot=/workspace/data/grasp-gold/raw/2wiki/dev.json \
  --gold-source hotpotqa_abstracts_2017=/workspace/data/grasp-gold/raw/hotpotqa/dev.json \
  --gold-source wiki18=/workspace/data/grasp-gold/raw/nq/dev.json \
  --output /workspace/data/grpo_dev_with_gold.jsonl --split dev
```

```powershell
git clone <your-project-b-repository> grasp-grpo-trainer
cd grasp-grpo-trainer
pip install -r requirements.txt
hf auth login
hf download doancuavahk/grasp-qwen2.5-3b-sft --local-dir /workspace/checkpoints/grasp-qwen2.5-3b-sft
hf download doancuavahk/grasp-retrieval --repo-type dataset --local-dir /workspace/data/grasp-retrieval
```

Start a frozen OpenAI-compatible Qwen2.5-7B reader endpoint, then validate artifacts:

```powershell
grasp-grpo-validate --checkpoint /workspace/checkpoints/grasp-qwen2.5-3b-sft `
  --index-dir /workspace/data/grasp-retrieval/hotpotqa_abstracts_2017 `
  --triple-cache /workspace/data/grasp-retrieval/cache/triples.sqlite
```

## Smoke rollout

Use one question before training. The result is audit data only: it must never be reused after
the actor receives an update.

```powershell
grasp-grpo-rollout --checkpoint /workspace/checkpoints/grasp-qwen2.5-3b-sft `
  --qa /workspace/data/train.jsonl `
  --index-dir /workspace/data/grasp-retrieval/hotpotqa_abstracts_2017 `
  --triple-cache /workspace/data/grasp-retrieval/cache/triples.sqlite `
  --reader-model Qwen/Qwen2.5-7B-Instruct --output outputs/smoke-tree.json --limit 1
```

## Train GRPO

```powershell
grasp-train-grpo --sft-checkpoint /workspace/checkpoints/grasp-qwen2.5-3b-sft `
  --qa /workspace/data/train.jsonl `
  --index-dir /workspace/data/grasp-retrieval/hotpotqa_abstracts_2017 `
  --triple-cache /workspace/data/grasp-retrieval/cache/triples.sqlite `
  --reader-model Qwen/Qwen2.5-7B-Instruct `
  --output-dir /workspace/outputs/grpo --iterations 100
```

For one mixed training file spanning all three corpora, provide `--corpus-config` instead of
`--index-dir` and `--triple-cache`. Its keys must exactly equal each example's `dataset` field:

`config/corpora.vast.json` is that mapping for the Hugging Face layout.

For every iteration, the trainer samples a fresh tree using the current actor, scores all leaves,
normalizes rewards within that question/tree, then applies one clipped GRPO update plus KL to the
frozen SFT reference. The implementation fixes branching to two, defaults to depth 3 and caps the
tree at four leaves.

To continue from an interrupted run, keep `--sft-checkpoint` pointed at the original Project A
adapter and supply the last GRPO adapter separately with `--actor-checkpoint`. This preserves the
intended fixed KL reference.

## Important limitations and operational choices

- The Project A adapter must be downloaded locally first; this project intentionally does not
  accept an HF repository ID as a checkpoint path.
- The actor and the frozen KL reference are two adapters on one base model (bf16 by default, about
  6 GB; `LOAD_IN_4BIT=1` for nf4). The backward pass uses gradient checkpointing with dropout off,
  and only response positions get logits, so a 24 GB card holds the trainer.
- `grasp-grpo-chat` is deliberately a basic reader/retriever demo. Production evaluation must use
  the same datasets, corpus, reader, context budget and metric implementation for GRASP and every
  baseline.
- The binary tree is a training-time exploration structure. A production chatbot normally performs
  a single sequential trajectory to control latency.

## Baseline comparison

Make every system write `[{"id": "...", "prediction": "..."}]` against the same QA file.
This small shared evaluator prevents comparing systems on mismatched questions:

```powershell
grasp-evaluate --qa data/hotpotqa_test.jsonl --predictions outputs/grasp_predictions.json `
  --output outputs/grasp_metrics.json
grasp-evaluate --qa data/hotpotqa_test.jsonl --predictions outputs/dense_rag_predictions.json `
  --output outputs/dense_rag_metrics.json
```

For paper-grade comparison, retain the benchmark-specific evaluator for ASQA ROUGE and any official
HotPotQA/TriviaQA normalization; this generic evaluator reports F1 and EM only.

## Checkpoint evaluation

Use deterministic sequential inference for checkpoint selection. Gold document IDs are read only by
the evaluator after predictions are produced; they are never exposed to the proxy or reader.

```bash
CUDA_VISIBLE_DEVICES=1 grasp-infer \
  --checkpoint /workspace/checkpoints/grasp-qwen2.5-3b-sft \
  --qa /workspace/data/grasp-gold-evidence/dev.jsonl \
  --corpus-config /workspace/configs/corpora.json \
  --reader-url http://127.0.0.1:8000/v1 --reader-api-key "$READER_API_KEY" \
  --reader-model Qwen/Qwen2.5-7B-Instruct \
  --output /workspace/outputs/eval/sft-dev-predictions.json --load-in-4bit
grasp-evaluate --qa /workspace/data/grasp-gold-evidence/dev.jsonl \
  --predictions /workspace/outputs/eval/sft-dev-predictions.json \
  --output /workspace/outputs/eval/sft-dev-metrics.json
```

Repeat with each GRPO checkpoint and select by dev F1 and context recall before touching the test
split.

## Training protocol (stages 0–4)

Every GPU hour is gated; see `docs/superpowers/specs/2026-11-04-grpo-training-protocol-design.md`.

| Stage | Script | Passes when |
|---|---|---|
| 0 | `scripts/stage0_baseline.sh` | SFT F1 ≥ 0.30 with the Project A reader, ≤ 50% zero-signal trees, ≤ 20 s per tree |
| 1 | `scripts/stage1_smoke.sh` | finite loss, ratio ≈ 1, 0 < KL < 0.05, non-zero gradients, LoRA weights moved |
| 2 | `scripts/stage2_overfit.sh` | reward on 8 fixed questions rises ≥ 30% in 40 steps |
| 3 | `scripts/stage3_pilot.sh` | eval F1 ≥ SFT + 0.02 and no early-stop rule fired |
| 4 | `scripts/train_grpo_vast.sh` | early stopping decides; use `best/` |

Each stage writes `$OUTPUT_ROOT/stage<N>/stage_report.json` (`passed`, `reasons`). Do not start a stage before the previous one passed.
Fixed eval ids live in `config/eval_ids.json` (`grasp-make-eval-ids`). Training stops itself on: no eval gain ≥ 0.02 F1 for 3 evals, F1 ≥ 0.03 below SFT twice, format rate < 90%, token KL > 0.3, entropy < 30% of start, > 60% zero-signal trees, or `--max-hours`.
