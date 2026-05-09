# Guia de Avaliação — CLASP Retrieval

Este guia cobre como rodar a avaliação de retrieval (candidate e matrix mode) e a avaliação de robustez a ruído para cada dataset suportado: **VoxPopuli**, **Spoken SQuAD** e **SPIRAL**.

---

## Modos de avaliação

| Modo | Script | O que faz |
|------|--------|-----------|
| `candidate` | `run_retrieval_eval.py` | Amostras negativas aleatórias do PKL; usa `--model-path` |
| `matrix` | `run_retrieval_eval.py` | Matriz full de similaridade no split `test`; requer `clasp_emb` pré-computado |
| `spiral` | `run_retrieval_eval.py` | On-the-fly embeddings a partir de JSONL; usa `--model-path` |
| noise | `run_noise_robustness_eval.py` | Adiciona ruído ao áudio de teste; compara Hits@1/MRR com/sem ruído |

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

Requer `total_dataset['test']['clasp_emb']` pré-computado (rode o forward no split de test primeiro):

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

O PKL de VoxPopuli inclui `audio_path` no split `test`; passe `--audio-paths-from-pickle`.
Use `--num-candidates` igual ao tamanho do split de test (dataset inteiro):

```bash
# obter tamanho do split
N=$(python -c "import pickle; d=pickle.load(open('data/datasets/total_dataset_voxpopuli.pkl','rb')); print(len(d['test']['text']))")

python scripts/run_noise_robustness_eval.py \
  --dataset-path data/datasets/total_dataset_voxpopuli.pkl \
  --model-path models/checkpoints/clasp_voxpopuli.pt \
  --audio-paths-from-pickle \
  --num-candidates "$N" \
  --snr-levels 20,15,10,5 \
  --output-csv results/noise_voxpopuli.csv
```

Para incluir ruído ambiente (WHAM), adicione `--wham-dir /path/to/wham_noise`.

---

## 2. Spoken SQuAD

### Candidate mode

```bash
# obter tamanho do split (validation, pois Spoken SQuAD não tem split 'test')
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

```bash
# obter tamanho do split
N=$(python -c "import pickle; d=pickle.load(open('data/datasets/total_dataset_spoken_squad.pkl','rb')); k='test' if 'test' in d else 'validation'; print(len(d[k]['text']))")

python scripts/run_noise_robustness_eval.py \
  --dataset-path data/datasets/total_dataset_spoken_squad.pkl \
  --model-path models/checkpoints/clasp_spoken_squad.pt \
  --wav-dir dev_wav/ \
  --train-json spoken_test-v1.1.json \
  --num-candidates "$N" \
  --snr-levels 20,15,10,5 \
  --output-csv results/noise_spoken_squad.csv
```

---

## 3. SPIRAL

O SPIRAL roda diretamente do JSONL sem PKL separado:

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

Flags adicionais:

```bash
  --spiral-audio-pooling mean      # ou max_sim (ColBERT-style)
  --spiral-chunk-samples 320000    # janela HuBERT (~20s a 16kHz)
  --max-samples 500                # limita amostras (debug)
```

---

## Flags comuns de avaliação

### `run_retrieval_eval.py`

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--mode` | `candidate` | `candidate`, `matrix` ou `spiral` |
| `--dataset-path` | — | PKL (candidate/matrix) ou JSONL (spiral) |
| `--model-path` | — | Checkpoint CLASP (obrigatório em candidate e spiral) |
| `--audio-key` | `hubert-emb` | Chave de áudio no PKL |
| `--text-key` | `text` | Chave de texto no PKL |
| `--emb-key` | `clasp_emb` | Chave de embedding para matrix mode |
| `--threshold` | `0.5` | Limiar de similaridade |
| `--num-candidates` | `100` | Candidatos negativos por query; use o tamanho do split inteiro para eval completo |
| `--hits-k` | `1,5,10,50` | K values para Hits@K |
| `--plot-out` | `None` | Salva PNG com métricas (matrix mode) |
| `--by-source` | — | Métricas por fonte (matrix mode) |

### `run_noise_robustness_eval.py`

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--dataset-path` | — | PKL com split `test` |
| `--model-path` | — | Checkpoint CLASP |
| `--audio-paths-from-pickle` | — | Usa `test['audio_path']` do PKL |
| `--wav-dir` | — | Pasta com WAVs (alternativa ao pickle) |
| `--train-json` | — | JSON do Spoken SQuAD para reconstruir paths |
| `--snr-levels` | `20,15,10,5` | Níveis de SNR em dB |
| `--num-candidates` | `10` | Candidatos por query; use o tamanho do split inteiro para eval completo |
| `--output-csv` | `None` | Salva resultados em CSV |
| `--wham-dir` | `None` | Dir. com ruído ambiente WHAM |
