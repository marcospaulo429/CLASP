# MSEB / SVQ Evaluation Guide

This guide covers evaluating CLASP on the [Massive Sound Embedding Benchmark
(MSEB)](https://github.com/google-research/mseb) using the [Simple Voice
Questions (SVQ)](https://huggingface.co/datasets/google/svq) dataset:

- **Acoustic Hypothesis Reranking** — for each spoken query, reorder a fixed set
  of phonetically-confusable text candidates by their relevance to what was
  actually said. Metrics: **MAP** (main), **MRR@10**, **WER**, **CER**.
- **Audio→text retrieval** — SVQ document / passage retrieval (in-lang and
  cross-lang), confirming CLASP retrieval works on the new dataset.

CLASP plugs in as an `mseb.encoder.MultiModalEncoder`
(`src/clasp/mseb_adapter/clasp_encoder.py`): a `Sound` becomes the 768-d fused
CLASP audio embedding (HuBERT + EfficientNet-B7 spectrogram-image), a `Text`
becomes its 768-d LaBSE embedding, and both are L2-normalized so MSEB's default
`dot_product` distance equals CLASP's training cosine.

## Environment

**MSEB requires Python ≥ 3.12** (the base CLASP env is 3.10), so use a dedicated
environment. CLASP itself supports 3.12, so both coexist there.

```bash
# dedicated 3.12 env for MSEB runs
uv venv --python 3.12 .venv-mseb
VIRTUAL_ENV=.venv-mseb uv pip install -e ".[mseb]"
# CLASP runtime deps used by the adapter (if not already pulled in):
VIRTUAL_ENV=.venv-mseb uv pip install sentence-transformers torchvision matplotlib
```

The `[mseb]` extra pins `openai-whisper` on purpose: the reranking evaluator
imports `whisper.normalizers`, which the unrelated `whisper` PyPI package lacks.

### Platform note

On Apple Silicon (ARM macOS) the SVQ dataset loader (`array_record` /
`apache_beam`) **segfaults** when co-loaded with torch, so full task runs must
happen on **Linux/x86** or via the repo's Docker images (`docker/`). Retrieval
tasks additionally require `scann`, which only ships x86-Linux wheels. Reranking
task *modules* import fine on macOS; only the dataset I/O is affected.

## Running a task

```bash
.venv-mseb/bin/python scripts/run_mseb_task.py \
  --task SVQEnUsQueryReranking \
  --model-path models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt \
  --results-jsonl artifacts_user/svq_rerank_enus.jsonl
```

The driver imports only the specific task module needed (avoiding MSEB's heavy
all-tasks import), builds a `DirectRunner` around the CLASP encoder, runs the
task, and prints/saves the `LeaderboardResult` JSON.

Batch wrapper (writes logs + per-task JSONL under `logs/` and `artifacts_user/`):

```bash
bash experiments/run_svq_mseb.sh \
  models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt \
  SVQEnUsQueryReranking SVQEnGbQueryReranking
```

### Driver flags (`scripts/run_mseb_task.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | — | MSEB task name (see below) |
| `--model-path` | — | CLASP fusion checkpoint (`.pt`) |
| `--results-jsonl` | `None` | Write `LeaderboardResult` JSON here |
| `--device` | auto | torch device (`cuda` / `cpu` / `mps`) |
| `--batch-size` | 1 | Encoder batch size |
| `--num-threads` | 1 | Runner threads |
| `--chunk-batch-size` | 4 | HuBERT / EfficientNet chunk batch size |
| `--task-module` | guessed | Dotted module to import so the task registers |
| `--encoder-name` | `clasp` | Label recorded in the results |

## Task names

- **Reranking** (26 locales): `SVQ<Locale>QueryReranking` — e.g.
  `SVQEnUsQueryReranking`, `SVQEnGbQueryReranking`, `SVQArEgQueryReranking`,
  `SVQJaJpQueryReranking`, `SVQHiInQueryReranking`, …
- **Retrieval** (Linux/x86): `SVQ<Locale>DocumentInLangRetrieval`,
  `SVQ<Locale>PassageInLangRetrieval`, and the `...CrossLang...` variants —
  e.g. `SVQEnUsDocumentInLangRetrieval`.

Compare CLASP's numbers against the baselines under `mseb/results/*` in the MSEB
repo and the [leaderboard](https://google-research.github.io/mseb/leaderboard.html).

---

## End-to-end RAG on SVQ (native harness)

`scripts/run_svq_rag_eval.py` runs a full **retrieve → generate → score** pipeline on
SVQ, independent of the MSEB package (so it runs in the base Python 3.10 env, on a
CUDA GPU such as a 4090):

1. **Retrieve** — CLASP encodes each spoken query and retrieves the top-k passages
   (cosine vs. LaBSE passage embeddings) from the SVQ reasoning corpus.
2. **Generate** — a local LLM (default **Qwen3-8B**) answers using those passages.
3. **Score** — EM / token-F1 vs. the SVQ gold answer span(s), plus retrieval
   **Recall@k** so retrieval and generation quality are separable.

The question text handed to the LLM is SVQ's gold transcript, so the score isolates
CLASP retrieval + LLM generation (no ASR error). SVQ reasoning rows carry `utt_id` but
not audio, so the harness **joins** them to the waveform by `utt_id` from the `audio`
config (streamed — no full download).

```bash
# install the extra (base 3.10 env; needs a CUDA GPU for the default generator)
uv pip install -e ".[rag]"

python scripts/run_svq_rag_eval.py \
  --model-path models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt \
  --config span_reasoning_in_lang --locale en_us \
  --generator Qwen/Qwen3-8B --top-k 5 \
  --output-json artifacts_user/svq_rag_en_us.json
```

`--dry-run-generator` skips the LLM (retrieval + scoring only) for a quick pipeline
check. Wrapper: `bash experiments/run_svq_rag.sh <MODEL> [LOCALE] [MAX_SAMPLES]`
(`DRY_RUN=1` and `GENERATOR=<hf-id>` env overrides supported).

### RAG flags (`scripts/run_svq_rag_eval.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | — | CLASP fusion checkpoint |
| `--config` | `span_reasoning_in_lang` | SVQ reasoning config (question + passage + gold span) |
| `--locale` | `en_us` | Locale filter (`''` = all) |
| `--audio-config` | `audio` | SVQ config with the waveforms, joined by `utt_id` (streamed) |
| `--top-k` | `5` | Passages retrieved and fed to the LLM |
| `--generator` | `Qwen/Qwen3-8B` | HF model id for the answer LLM |
| `--dry-run-generator` | — | Stub the LLM (retrieval + scoring only) |
| `--max-new-tokens` | `64` | Generation length cap |
| `--enable-thinking` | — | Qwen3 thinking mode (off by default) |
| `--max-samples` | all | Cap the number of questions |
| `--output-json` | `None` | Write summary + per-row results |
