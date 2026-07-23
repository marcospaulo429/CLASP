# Training Guide — CLASP

This guide covers building the dataset pickle and launching training for each of
the three supported data sources: **VoxPopuli**, **Spoken SQuAD**, and **SPIRAL**.

---

## Prerequisites

```bash
# base environment
uv sync

# per-dataset extras
uv sync --extra voxpopuli     # VoxPopuli
uv sync --extra realdata      # Spoken SQuAD / SpeechBrown
```

Official checkpoint for warm-start (recommended):

```bash
huggingface-cli download llm-lab/CLASP CLASP_Concat_Final_Fusion_Encoder.pt \
  --local-dir models/checkpoints
```

---

## 1. VoxPopuli (English)

### 1a. Build the dataset

```bash
# HF validation split (default) — creates a PKL with train/validation/test splits
python scripts/build_voxpopuli_pkl.py \
  --output data/datasets/total_dataset_voxpopuli.pkl \
  --audio-cache-dir data/datasets/voxpopuli_en_validation_wav \
  --replicate-for-train \
  --max-samples 2000          # remove to use everything (~1752 samples)
```

Useful flags:
- `--require-gold-only` — use only transcripts marked as gold
- `--val-fraction 0.1` — 90% train / 10% validation (instead of `--replicate-for-train`)
- `--hf-split train` — use the HF train split (much larger; downloads train parquets)
- `--validation-parquet /path/to/file.parquet` — for offline use

### 1b. Training

```bash
python scripts/train.py \
  --dataset-path data/datasets/total_dataset_voxpopuli.pkl \
  --save-path models/checkpoints/clasp_voxpopuli.pt \
  --init-checkpoint models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt \
  --num-epochs 50 \
  --learning-rate 1e-4 \
  --batch-size-train 32 \
  --batch-size-val 16
```

### 1c. All-in-one script (build + train)

```bash
bash scripts/run_training.sh [MAX_TRAIN_SAMPLES] [MAX_VAL_SAMPLES] [NUM_EPOCHS]

# Examples:
bash scripts/run_training.sh                 # defaults: 10000 / all / 50 epochs
bash scripts/run_training.sh 50000 1000 100  # 50k train, 1k val, 100 epochs
bash scripts/run_training.sh all all 100     # full dataset
```

---

## 2. Spoken SQuAD

The JSON files (`spoken_train-v1.1.json`, `spoken_test-v1.1.json`) live at the
project root. Train WAVs go in `train_wav/` and dev WAVs in `dev_wav/`.

### 2a. Build the dataset

```bash
python scripts/build_spoken_squad_pkl.py \
  --train-json spoken_train-v1.1.json \
  --train-wav-dir train_wav/ \
  --val-json spoken_test-v1.1.json \
  --val-wav-dir dev_wav/ \
  --output data/datasets/total_dataset_spoken_squad.pkl
```

For a quick test (limit samples):

```bash
python scripts/build_spoken_squad_pkl.py \
  --train-json spoken_train-v1.1.json \
  --train-wav-dir train_wav/ \
  --val-json spoken_test-v1.1.json \
  --val-wav-dir dev_wav/ \
  --output data/datasets/total_dataset_spoken_squad.pkl \
  --max-train-samples 200 \
  --max-val-samples 50
```

### 2b. Training

```bash
python scripts/train.py \
  --dataset-path data/datasets/total_dataset_spoken_squad.pkl \
  --save-path models/checkpoints/clasp_spoken_squad.pt \
  --init-checkpoint models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt \
  --num-epochs 50 \
  --learning-rate 1e-4 \
  --batch-size-train 32 \
  --batch-size-val 16
```

---

## 3. SPIRAL

SPIRAL uses on-the-fly evaluation directly from the JSONL — no separate PKL build
is needed. Training with SPIRAL still requires a PKL (use VoxPopuli or Spoken
SQuAD as the base).

To **evaluate** on SPIRAL, see the [evaluation guide](EVAL.md#3-spiral).

---

## Common `train.py` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-path` | — | PKL with `train` / `validation` splits |
| `--save-path` | — | Where to save the trained checkpoint |
| `--init-checkpoint` | `None` | Fusion weights for warm-start (does not restore the optimizer) |
| `--num-epochs` | 100 | Number of epochs |
| `--learning-rate` | 1e-4 | Learning rate |
| `--batch-size-train` | 32 | Training batch size |
| `--batch-size-val` | 16 | Validation batch size |
| `--patience` | 10 | Early stopping: epochs without improvement |
| `--no-early-stopping` | — | Disable early stopping |
| `--freeze-encoders` | — | Freeze `audio_seq` and `image_seq`; train only the fusion head |
| `--wandb-project` | `None` | Enable W&B logging |
| `--in-features-text` | 1024 | Audio (HuBERT) input dim of the encoder |
| `--in-features-image` | 1000 | Image (EfficientNet-B7) input dim of the encoder |

---

## Docker (VoxPopuli)

```bash
docker run --rm -it --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN \
  -w /app clasp:amd64 \
  python scripts/build_voxpopuli_pkl.py \
    --replicate-for-train --max-samples 2000 \
    --output data/datasets/total_dataset_voxpopuli.pkl

docker run --rm -it --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -w /app clasp:amd64 \
  python scripts/train.py \
    --dataset-path data/datasets/total_dataset_voxpopuli.pkl \
    --save-path models/checkpoints/clasp_vox.pt \
    --init-checkpoint models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt \
    --num-epochs 50
```

Use `clasp:arm64` on Apple Silicon / ARM machines.
