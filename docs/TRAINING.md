# Guia de Treino — CLASP

Este guia cobre como construir o dataset pickle e disparar o treino para cada uma das três fontes de dados suportadas: **VoxPopuli**, **Spoken SQuAD** e **SPIRAL**.

---

## Pré-requisitos

```bash
# Ambiente base
uv sync

# Extras por dataset
uv sync --extra voxpopuli     # VoxPopuli
uv sync --extra realdata      # Spoken SQuAD / SpeechBrown
```

Checkpoint oficial para warm-start (recomendado):

```bash
huggingface-cli download llm-lab/CLASP CLASP_Concat_Final_Fusion_Encoder.pt \
  --local-dir models/checkpoints
```

---

## 1. VoxPopuli (English)

### 1a. Build do dataset

```bash
# Split de validação do HF (padrão) — cria PKL com splits train/validation/test
python scripts/build_voxpopuli_pkl.py \
  --output data/datasets/total_dataset_voxpopuli.pkl \
  --audio-cache-dir data/datasets/voxpopuli_en_validation_wav \
  --replicate-for-train \
  --max-samples 2000          # remova para usar tudo (~1752 amostras)
```

Flags úteis:
- `--require-gold-only` — usa somente transcrições marcadas como gold
- `--val-fraction 0.1` — 90 % treino / 10 % validação (em vez de `--replicate-for-train`)
- `--hf-split train` — usa o split de treino do HF (muito maior, baixa parquets de treino)
- `--validation-parquet /path/to/file.parquet` — para uso offline

### 1b. Treino

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

### 1c. Script all-in-one (build + treino)

```bash
bash scripts/run_training.sh [MAX_TRAIN_SAMPLES] [MAX_VAL_SAMPLES] [NUM_EPOCHS]

# Exemplos:
bash scripts/run_training.sh                 # defaults: 10000 / todos / 50 epochs
bash scripts/run_training.sh 50000 1000 100  # 50k treino, 1k val, 100 epochs
bash scripts/run_training.sh all all 100     # dataset completo
```

---

## 2. Spoken SQuAD

Os arquivos JSON (`spoken_train-v1.1.json`, `spoken_test-v1.1.json`) ficam na raiz do projeto.
Os WAVs de treino ficam em `train_wav/` e os de dev em `dev_wav/`.

### 2a. Build do dataset

```bash
python scripts/build_spoken_squad_pkl.py \
  --train-json spoken_train-v1.1.json \
  --train-wav-dir train_wav/ \
  --val-json spoken_test-v1.1.json \
  --val-wav-dir dev_wav/ \
  --output data/datasets/total_dataset_spoken_squad.pkl
```

Para teste rápido (limita amostras):

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

### 2b. Treino

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

O SPIRAL usa avaliação on-the-fly diretamente do JSONL — não precisa de build de PKL separado.
O treino com SPIRAL ainda requer um PKL (use VoxPopuli ou Spoken SQuAD como base).

Para **avaliar** no SPIRAL, veja o [guia de avaliação](EVAL.md#3-spiral).

---

## Flags comuns do `train.py`

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--dataset-path` | — | PKL com splits `train` / `validation` |
| `--save-path` | — | Onde salvar o checkpoint treinado |
| `--init-checkpoint` | `None` | Fusion weights para warm-start (não restaura otimizador) |
| `--num-epochs` | 100 | Número de epochs |
| `--learning-rate` | 1e-4 | Taxa de aprendizado |
| `--batch-size-train` | 32 | Batch de treino |
| `--batch-size-val` | 16 | Batch de validação |
| `--patience` | 10 | Early stopping: epochs sem melhora |
| `--no-early-stopping` | — | Desativa early stopping |
| `--freeze-encoders` | — | Congela `audio_seq` e `image_seq`; treina só o fusion head |
| `--wandb-project` | `None` | Habilita logging no W&B |
| `--in-features-text` | 1024 | Dim. de entrada do encoder de texto (LaBSE = 1024) |
| `--in-features-image` | 1000 | Dim. de entrada do encoder de imagem (EfficientNet-B7 = 1000) |

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

Use `clasp:arm64` em máquinas Apple Silicon / ARM.
