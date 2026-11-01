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

`READER_GPU=0` and `TRAIN_GPU=1` target a single two-GPU Vast instance. Set `READER_URL` to the
reader instance's mapped public URL only when the reader is hosted on a separate Vast instance.
The setup script creates a separate `READER_VENV` for vLLM because its dependency versions conflict
with the stable Transformers/PEFT stack used by the GRPO trainer.

Build the three compatible SQLite triple caches from the downloaded teacher trajectories before
training: `bash scripts/build_triple_caches_vast.sh`. Point the corpus config's `triple_cache`
entries at `/workspace/data/grasp-triples/{2wiki,hotpotqa,wiki18}.sqlite`.

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
  --index-dir /workspace/data/grasp-retrieval/hotpotqa_ircot `
  --triple-cache /workspace/data/grasp-retrieval/hotpotqa_ircot/triples.sqlite
```

## Smoke rollout

Use one question before training. The result is audit data only: it must never be reused after
the actor receives an update.

```powershell
grasp-grpo-rollout --checkpoint /workspace/checkpoints/grasp-qwen2.5-3b-sft `
  --qa /workspace/data/train.jsonl `
  --index-dir /workspace/data/grasp-retrieval/hotpotqa_ircot `
  --triple-cache /workspace/data/grasp-retrieval/hotpotqa_ircot/triples.sqlite `
  --reader-model Qwen/Qwen2.5-7B-Instruct --output outputs/smoke-tree.json --limit 1
```

## Train GRPO

```powershell
grasp-train-grpo --sft-checkpoint /workspace/checkpoints/grasp-qwen2.5-3b-sft `
  --qa /workspace/data/train.jsonl `
  --index-dir /workspace/data/grasp-retrieval/hotpotqa_ircot `
  --triple-cache /workspace/data/grasp-retrieval/hotpotqa_ircot/triples.sqlite `
  --reader-model Qwen/Qwen2.5-7B-Instruct `
  --output-dir /workspace/outputs/grpo --iterations 100
```

For one mixed training file spanning all three corpora, provide `--corpus-config` instead of
`--index-dir` and `--triple-cache`. Its keys must exactly equal each example's `dataset` field:

```json
{
  "hotpotqa_ircot": {"index_dir": "/workspace/data/grasp-retrieval/hotpotqa_ircot", "triple_cache": "/workspace/data/grasp-retrieval/hotpotqa_ircot/triples.sqlite"},
  "2wiki_ircot": {"index_dir": "/workspace/data/grasp-retrieval/2wiki_ircot", "triple_cache": "/workspace/data/grasp-retrieval/2wiki_ircot/triples.sqlite"},
  "wiki18_ircot": {"index_dir": "/workspace/data/grasp-retrieval/wiki18_ircot", "triple_cache": "/workspace/data/grasp-retrieval/wiki18_ircot/triples.sqlite"}
}
```

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
- A 24 GB GPU is typically suited to QLoRA. Do not load actor and reference in 4-bit unless your
  bitsandbytes/PEFT stack supports training the exact adapter configuration.
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
