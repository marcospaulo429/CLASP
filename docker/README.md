# Docker images

| File | Architecture | Compose service |
|------|----------------|-----------------|
| [`Dockerfile.amd64`](Dockerfile.amd64) | `linux/amd64` (x86_64) | `clasp-amd64` → image `clasp:amd64` |
| [`Dockerfile.arm64`](Dockerfile.arm64) | `linux/arm64` (Apple Silicon, ARM servers) | `clasp-arm64` → image `clasp:arm64` |

Build from the **repository root** (context must be `.` so `COPY pyproject.toml`, `src/`, and `scripts/` resolve):

```bash
docker compose build clasp-amd64   # x86_64
docker compose build clasp-arm64   # ARM64
```

Optional dependency group (LaBSE, datasets, etc.) is set in [`docker-compose.yml`](../docker-compose.yml) as `UV_EXTRA_GROUPS=voxpopuli`. For a lighter SPIRAL-only image, override with `realdata`:

```bash
docker build -f docker/Dockerfile.amd64 -t clasp:spiral-eval \
  --build-arg UV_EXTRA_GROUPS=realdata .
```

## Do not mount the whole repo on `/app`

A bind mount like `-v "$PWD:/app"` **replaces** the image’s `/app`, including the **`.venv`** installed at build time (`torch`, `transformers`, etc.). You will see `ModuleNotFoundError: No module named 'torch'`.

**Do this instead:** keep the image’s `/app` and mount only what you need:

- `data/`, `models/`, `results/` for outputs
- `scripts/` and `src/` if you need **host** code newer than the image
- JSON + `train_wav` / `dev_wav` (or their parent dirs) as separate mounts

### Example: `build_spoken_squad_pkl.py` (train + dev)

From the repo root (adjust paths if your JSON/WAVs live elsewhere):

```bash
export REPO_ROOT="$HOME/CLASP"
mkdir -p "${REPO_ROOT}/data/datasets"

docker run --rm -it --gpus all \
  -w /app \
  -v "${REPO_ROOT}/data:/app/data" \
  -v "${REPO_ROOT}/scripts:/app/scripts" \
  -v "${REPO_ROOT}/src:/app/src" \
  -v "${REPO_ROOT}/train_wav:/app/train_wav" \
  -v "${REPO_ROOT}/dev_wav:/app/dev_wav" \
  -v "${REPO_ROOT}/spoken_train-v1.1.json:/app/spoken_train-v1.1.json:ro" \
  -v "${REPO_ROOT}/spoken_dev-v1.1.json:/app/spoken_dev-v1.1.json:ro" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN \
  clasp:arm64 \
  python scripts/build_spoken_squad_pkl.py \
    --train-json spoken_train-v1.1.json \
    --train-wav-dir train_wav \
    --dev-json spoken_dev-v1.1.json \
    --dev-wav-dir dev_wav \
    --output data/datasets/total_dataset_spoken_squad_trainval.pkl
```

Use `clasp:amd64` on x86_64 hosts. Remove `--gpus all` for CPU-only.

### Example: `train.py`

```bash
docker run --rm -it --gpus all \
  -w /app \
  -v "${REPO_ROOT}/data:/app/data" \
  -v "${REPO_ROOT}/models:/app/models" \
  -v "${REPO_ROOT}/scripts:/app/scripts" \
  -v "${REPO_ROOT}/src:/app/src" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  clasp:arm64 \
  python scripts/train.py \
    --dataset-path data/datasets/total_dataset_spoken_squad_trainval.pkl \
    --save-path models/checkpoints/meu_modelo.pt \
    --batch-size-train 32 \
    --batch-size-val 16
```

