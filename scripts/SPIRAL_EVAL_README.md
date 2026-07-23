# CLASP SPIRAL Long-Context Speech Retrieval Evaluation

Evaluation of CLASP on **SPIRAL** using the **same metrics stack** as the pickle
`matrix` mode (`Hits@K`, `MRR`, `mean_rank`, `median_rank`, tie-breaking in
`ranking_metrics.py`) and **shared plots** (`retrieval_plots.py`).

## Requirements

- `pip install -e .`
- `pip install torchvision sentence-transformers` (LaBSE via `SentenceTransformer.encode`, same as the pickle build scripts)
- SPIRAL audio: https://duke.box.com/v/spiral-dataset
- **Alternative**: Docker image with the `realdata` extra (see *Docker evaluation* below).

### Where to put the WAVs

The JSONL points at `audio_folder`: `spiral/wavs` (e.g. `spiral/wavs/lecture_0.wav`).
The loader also accepts these folders relative to `--spiral-audio-base` or to the
`data.jsonl` folder:

- `spiral/wavs/*.wav` (the "official" layout)
- `wavs/wavs/*.wav` (common after extracting the archive into `spiral_dataset/wavs/wavs/`)

Use `--spiral-audio-base` as the **parent** directory of those folders (usually the
`spiral_dataset` folder containing `data.jsonl` and `wavs/`).

## Recommended way: `run_retrieval_eval.py --mode spiral`

```bash
source venv/bin/activate

python scripts/run_retrieval_eval.py \
  --mode spiral \
  --dataset-path spiral_dataset/data.jsonl \
  --model-path models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt \
  --spiral-output-dir results/spiral \
  --spiral-audio-base spiral_dataset \
  --max-samples 50
```

Useful arguments:

| Flag | Description |
|------|-------------|
| `--spiral-audio-base` | Base directory used to resolve `spiral/wavs/...` |
| `--spiral-output-dir` | Output folder (plots + JSON) |
| `--max-samples` | Row limit for testing |
| `--batch-size-text` | Batch for `SentenceTransformer.encode` |
| `--hubert-model` | Default `facebook/hubert-large-ls960-ft` |
| `--sentence-transformer` | Default `sentence-transformers/LaBSE` |
| `--hits-k` | e.g. `1,5,10,50` (same as matrix mode) |
| `--spiral-audio-pooling` | `mean` (default): global average of chunks; `max_sim`: max-similarity per chunk (ColBERT-style) |
| `--spiral-chunk-samples` | Window size in 16 kHz samples (default `320000`); also used to map `key_sentence_timestamp` → chunk index in `max_sim` mode |

## Docker evaluation

The Dockerfiles live in `docker/` (`Dockerfile.amd64` for x86_64, `Dockerfile.arm64`
for ARM64). They install CLASP with `uv sync`. The SPIRAL eval needs **LaBSE** and
**torchvision** (extras `realdata` or `voxpopuli` in `pyproject.toml`). Build from the
**repo root** (context `.`):

```bash
# x86_64 (swap in docker/Dockerfile.arm64 on an ARM64 machine)
docker build -f docker/Dockerfile.amd64 -t clasp:spiral-eval \
  --build-arg UV_EXTRA_GROUPS=realdata \
  .
```

If your team already builds another tag (e.g. `clasp:arm64` / `clasp:amd64` from the
main README) with the same `UV_EXTRA_GROUPS`, use that image instead of
`clasp:spiral-eval`.

### `clasp:arm64` / `clasp:amd64` and the `invalid choice: 'spiral'` error

Older images only had `candidate` and `matrix`. **Rebuild** with the current code:

```bash
docker compose build clasp-arm64    # Apple Silicon / arm64
# on x86_64:
docker compose build clasp-amd64
```

The [`docker-compose.yml`](../docker-compose.yml) at the root installs the
**`voxpopuli`** extra (LaBSE, `torchvision`, etc.), which is compatible with the
SPIRAL eval.

**Without a rebuild**, you can mount the host's `scripts` and `src` to force the
latest code:

```text
-v "${REPO_ROOT}/scripts:/app/scripts" -v "${REPO_ROOT}/src:/app/src"
```

### Running the eval with the CLASP checkpoint (GPU)

Mount the dataset, the weights, the results folder, and (recommended) the Hugging
Face cache. Set `REPO_ROOT` to the absolute path of the checkout.

```bash
export REPO_ROOT="$(pwd)"

docker run --rm -it --gpus all \
  -w /app \
  -v "${REPO_ROOT}/spiral_dataset:/app/spiral_dataset" \
  -v "${REPO_ROOT}/models:/app/models" \
  -v "${REPO_ROOT}/results:/app/results" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN \
  clasp:spiral-eval \
  python scripts/run_retrieval_eval.py \
    --mode spiral \
    --dataset-path spiral_dataset/data.jsonl \
    --model-path models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt \
    --spiral-output-dir results/spiral \
    --spiral-audio-base spiral_dataset \
    --max-samples 50
```

- **Different checkpoint**: change `--model-path` to the desired `.pt` (the file must
  exist under `models/` on the host, mounted at `/app/models`).
- **CPU**: drop `--gpus all` (much slower).
- **Hub models** (HuBERT, LaBSE): downloaded on first run; the `~/.cache/huggingface`
  volume avoids re-downloads.

### Same pipeline via `run_spiral_retrieval.sh` in the container

```bash
export REPO_ROOT="/absolute/path/to/CLASP"

docker run --rm -it --gpus all -w /app \
  -v "${REPO_ROOT}/spiral_dataset:/app/spiral_dataset" \
  -v "${REPO_ROOT}/models:/app/models" \
  -v "${REPO_ROOT}/results:/app/results" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e SPIRAL_AUDIO_BASE=spiral_dataset \
  -e SPIRAL_OUTPUT_DIR=results/spiral \
  -e MAX_SAMPLES=50 \
  clasp:spiral-eval \
  bash scripts/run_spiral_retrieval.sh
```

Omitting `MODEL_PATH` uses the script default (checkpoint in `models/checkpoints/`).
For another `.pt`, use the absolute container path, e.g.
`-e MODEL_PATH=/app/models/checkpoints/other_name.pt`.

Extra arguments after the script (e.g. `--hits-k 1,5,10,50`) are forwarded to
`run_retrieval_eval.py`.

WAV details and `--spiral-audio-base` are documented in `spiral_dataset/wavs/README.md`.

## Automated pipeline (shell)

```bash
export SPIRAL_JSONL=/path/to/data.jsonl
export SPIRAL_AUDIO_BASE=/path/to/folder_with_spiral
export MODEL_PATH=/path/to/CLASP.pt
export SPIRAL_OUTPUT_DIR=results/spiral
./scripts/run_spiral_retrieval.sh
```

Optional variables: `MAX_SAMPLES`, `HUBERT_MODEL`, `SENTENCE_TRANSFORMER`. Extra
arguments are forwarded to `run_retrieval_eval.py`.

## Legacy wrapper

`scripts/eval_spiral_retrieval.py` just calls
`clasp.evaluation.spiral_runner.run_spiral_retrieval_eval` (same core).

## Python modules

- `clasp.data.spiral`: `load_spiral_jsonl`, `spiral_temporal_bin_indices`
- `clasp.evaluation.spiral_runner`: orchestration of embeddings + metrics + plots
- `clasp.evaluation.ranking_metrics`: `similarity_matrix_to_rows`, `compute_ranking_metrics`, `grouped_ranking_summary`
- `clasp.evaluation.retrieval_plots`: `save_retrieval_plot`, `save_grouped_hits_plot`

## Metrics

- Global: `Hits@K` (0–1 rate in the JSON; tables/console in %), `MRR`, `MAP`, `mean_rank`, `median_rank`.
- Per temporal bin: `Hits@1` per group (same 1-based rank criterion as the rest of the repo).

## Outputs (`--spiral-output-dir` folder)

| File | Content |
|------|---------|
| `retrieval_summary.png` | Hits@K bars + rank histogram (percentage mode) |
| `retrieval_by_temporal_bin.png` | Hits@1 per temporal bin + counts |
| `spiral_evaluation_results.json` | `ranking_metrics`, `grouped_temporal_bins`, metadata |

## Matrix mode: same global plot

```bash
python scripts/run_retrieval_eval.py --mode matrix --dataset-path total.pkl \
  --retrieval-plot-dir results/matrix_plots --hits-k 1,5,10,50
```

Generates `retrieval_summary.png` in that directory, using the same plot function
as SPIRAL.

## Mock demo

`scripts/eval_spiral_retrieval_demo.py` stays independent (synthetic data).

## SPIRAL citation

See the dataset README in `spiral_dataset/README.md`.
