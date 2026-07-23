# Evaluation Guide — CLASP Retrieval

This guide covers running retrieval evaluation (candidate and matrix modes) and
noise-robustness evaluation for each supported dataset: **VoxPopuli**,
**Spoken SQuAD**, and **SPIRAL**.

For the MSEB / SVQ reranking + retrieval tasks, see [MSEB.md](MSEB.md).

---

## Evaluation modes

| Mode | Script | What it does |
|------|--------|--------------|
| `candidate` | `run_retrieval_eval.py` | Random negatives sampled from the PKL; uses `--model-path` |
| `matrix` | `run_retrieval_eval.py` | Full similarity matrix on the `test` split; needs precomputed `clasp_emb` |
| `spiral` | `run_retrieval_eval.py` | On-the-fly embeddings from JSONL; uses `--model-path` |
| `paragraph_grouped` | `run_retrieval_eval.py` | Chunked PKL with `paragraph_id`; max-sim per paragraph |
| noise | `run_noise_robustness_eval.py` | Adds noise to test audio; compares Hits@K/MRR clean vs. noisy |

---

## 1. VoxPopuli

### Candidate mode

```bash
python scripts/run_retrieval_eval.py \
  --dataset-path data/datasets/total_dataset_voxpopuli.pkl \
  --mode candidate \
  --model-path models/checkpoints/clasp_voxpopuli.pt \
  --audio-key hubert-emb \
  --text-key text \
  --threshold 0.5 \
  --num-candidates 100
```

### Matrix mode

Requires precomputed `total_dataset['test']['clasp_emb']` (run the forward pass on
the test split first):

```bash
python scripts/run_retrieval_eval.py \
  --dataset-path data/datasets/total_dataset_voxpopuli.pkl \
  --mode matrix \
  --emb-key clasp_emb \
  --text-key text \
  --threshold 0.5 \
  --hits-k 1,5,10,50 \
  --plot-out artifacts/retrieval_voxpopuli.png
```

### Noise robustness

The VoxPopuli PKL stores a flat `audio_path` per row in the `test` split, which
the noise script auto-detects. See the [common flags](#noise-flags) below.

```bash
python scripts/run_noise_robustness_eval.py \
  --dataset-path data/datasets/total_dataset_voxpopuli.pkl \
  --model-path models/checkpoints/clasp_voxpopuli.pt \
  --snr-levels 20,15,10,5 \
  --noise-types white,reverb \
  --output-csv results/noise_voxpopuli.csv
```

For ambient noise, add `--esc50-dir /path/to/esc50` (or `--wham-dir /path/to/wham_noise`)
and include `ambient` in `--noise-types`.

---

## 2. Spoken SQuAD

### Candidate mode

```bash
# split size (Spoken SQuAD has no 'test' split, so fall back to 'validation')
N=$(python -c "import pickle; d=pickle.load(open('data/datasets/total_dataset_spoken_squad.pkl','rb')); k='test' if 'test' in d else 'validation'; print(len(d[k]['text']))")

python scripts/run_retrieval_eval.py \
  --dataset-path data/datasets/total_dataset_spoken_squad.pkl \
  --mode candidate \
  --model-path models/checkpoints/clasp_spoken_squad.pt \
  --audio-key hubert-emb \
  --text-key text \
  --threshold 0.5 \
  --num-candidates "$N"
```

### Matrix mode

```bash
python scripts/run_retrieval_eval.py \
  --dataset-path data/datasets/total_dataset_spoken_squad.pkl \
  --mode matrix \
  --emb-key clasp_emb \
  --text-key text \
  --threshold 0.5 \
  --hits-k 1,5,10,50 \
  --plot-out artifacts/retrieval_spoken_squad.png
```

### Noise robustness

The Spoken SQuAD PKL (built by `build_spoken_squad_pkl.py`) stores per-row
`audio_paths` plus a top-level `_meta.pooling_mode`. The noise script reads both:
`mean` pooling → candidate retrieval over the whole split;
`chunked` pooling → paragraph-grouped (max-sim) retrieval via `paragraph_id`.

```bash
python scripts/run_noise_robustness_eval.py \
  --dataset-path data/datasets/total_dataset_spoken_squad.pkl \
  --model-path models/checkpoints/clasp_spoken_squad.pt \
  --snr-levels 20,15,10,5 \
  --noise-types white,reverb \
  --output-csv results/noise_spoken_squad.csv
```

Ambient (ESC-50) example wrapper: `experiments/run_noise_ambient.sh <PKL> <MODEL>`.

---

## 3. SPIRAL

SPIRAL runs directly from a JSONL, without a separate PKL:

```bash
python scripts/run_retrieval_eval.py \
  --mode spiral \
  --dataset-path spiral_dataset/data.jsonl \
  --model-path models/checkpoints/clasp_voxpopuli.pt \
  --spiral-audio-base spiral_dataset/wavs/ \
  --spiral-output-dir results/spiral/ \
  --num-candidates 10 \
  --hits-k 1,5,10
```

Additional flags:

```bash
  --spiral-audio-pooling mean      # or max_sim (ColBERT-style)
  --spiral-chunk-samples 320000    # HuBERT window (~20s at 16kHz)
  --max-samples 500                # limit samples (debug)
```

See `scripts/SPIRAL_EVAL_README.md` for the full SPIRAL guide (Docker included).

---

## 4. SVQ (baseline retrieval)

Evaluate **baseline CLASP** (no additions) on the [SVQ](https://huggingface.co/datasets/google/svq)
dataset using the **same pipeline as the other datasets** — a `total_dataset` PKL
pairing each spoken query with its transcript (audio→text self-retrieval), scored by
`run_retrieval_eval.py`. This makes SVQ numbers directly comparable to VoxPopuli /
Spoken-SQuAD.

```bash
# 1) build the PKL (default config: audio_en_us_clean, ~1.4k clean English rows)
python scripts/build_svq_pkl.py \
  --config audio_en_us_clean \
  --output data/datasets/total_dataset_svq.pkl \
  --replicate-for-train

# 2) retrieval eval over the whole split
N=$(python -c "import pickle;d=pickle.load(open('data/datasets/total_dataset_svq.pkl','rb'));print(len(d['test']['text']))")
python scripts/run_retrieval_eval.py \
  --dataset-path data/datasets/total_dataset_svq.pkl \
  --mode candidate \
  --model-path models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt \
  --audio-key hubert-emb --text-key text \
  --num-candidates "$N"
```

Or in one step: `bash experiments/run_svq_retrieval.sh <MODEL> [CONFIG] [MAX_SAMPLES]`.
Use `--config audio --locale <loc>` (with `--max-samples`) for other locales.

For the MSEB acoustic-hypothesis **reranking** task and the **end-to-end RAG** eval on
SVQ, see [MSEB.md](MSEB.md).

---

## Common evaluation flags

### `run_retrieval_eval.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `candidate` | `candidate`, `matrix`, `spiral`, or `paragraph_grouped` |
| `--dataset-path` | — | PKL (candidate/matrix) or JSONL (spiral) |
| `--model-path` | — | CLASP checkpoint (required for candidate and spiral) |
| `--audio-key` | `hubert-emb` | Audio key in the PKL |
| `--text-key` | `text` | Text key in the PKL |
| `--emb-key` | `clasp_emb` | Embedding key for matrix mode |
| `--threshold` | `0.5` | Similarity threshold |
| `--num-candidates` | `100` | Negatives per query; use the full split size for a complete eval |
| `--hits-k` | `1,5,10,50` | K values for Hits@K |
| `--plot-out` | `None` | Save a metrics PNG (matrix mode) |
| `--by-source` | — | Per-source metrics (matrix mode) |

### <a name="noise-flags"></a>`run_noise_robustness_eval.py`

Reads a PKL built by `build_spoken_squad_pkl.py` (with `audio_paths`, optional
`paragraph_id`, and `_meta.pooling_mode`) or a VoxPopuli PKL (with `audio_path`).
It applies noise to the waveform, **re-extracts** HuBERT + EfficientNet
embeddings, and re-runs retrieval against the clean baseline.

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-path` | — | PKL with a `test`/`validation` split and `audio_paths`/`audio_path` |
| `--model-path` | — | CLASP checkpoint |
| `--audio-key` | `hubert-emb` | Audio embedding key |
| `--text-key` | `text` | Text embedding key |
| `--snr-levels` | `20,15,10,5` | SNR levels in dB (comma-separated) |
| `--noise-types` | `white,reverb` | Subset of `{white, ambient, reverb}` |
| `--wham-dir` | `None` | WHAM dir with `noise/*.wav` (ambient source) |
| `--esc50-dir` | `None` | ESC-50 dir with `audio/*.wav` (ambient source) |
| `--ambient-num-samples` | `64` | Ambient WAVs pre-loaded into the sampling pool |
| `--ambient-seed` | `42` | Seed for per-row ambient noise selection |
| `--chunk-samples` | `320000` | HuBERT window (~20s at 16kHz) |
| `--chunk-batch-size` | `4` | Chunk batch size for HuBERT/EfficientNet |
| `--retrieval-batch-size` | `64` | Fusion batch size for `paragraph_grouped` eval |
| `--hits-k` | `1,5,10,50` | K values for Hits@K |
| `--hubert-model` | `facebook/hubert-large-ls960-ft` | HuBERT model id |
| `--device` | auto | torch device |
| `--output-csv` | `None` | Save results (one row per `noise_config`) |

> The retrieval mode is derived automatically from `_meta.pooling_mode`
> (`chunked` → paragraph max-sim; otherwise candidate over the whole split).
> The older `--audio-paths-from-pickle`, `--wav-dir`, `--train-json`, and
> `--num-candidates` flags no longer exist.
