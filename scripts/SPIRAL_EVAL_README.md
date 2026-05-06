# CLASP SPIRAL Long-Context Speech Retrieval Evaluation

Avaliação do CLASP no **SPIRAL** usando a **mesma pilha de métricas** que o modo `matrix` do pickle (`Hits@K`, `MRR`, `mean_rank`, `median_rank`, tie-breaking em `ranking_metrics.py`) e **plots compartilhados** (`retrieval_plots.py`).

## Requisitos

- `pip install -e .`
- `pip install torchvision sentence-transformers` (LaBSE via `SentenceTransformer.encode`, igual aos scripts de build de pickle)
- Áudios SPIRAL: https://duke.box.com/v/spiral-dataset
- **Alternativa**: imagem Docker com extra `realdata` (seção *Avaliação com Docker* abaixo).

### Onde colocar os WAVs

O JSONL aponta para `audio_folder`: `spiral/wavs` (ex.: `spiral/wavs/lecture_0.wav`). O carregador também aceita estas pastas relativas a `--spiral-audio-base` ou à pasta do `data.jsonl`:

- `spiral/wavs/*.wav` (layout “oficial”)
- `wavs/wavs/*.wav` (comum após extrair o arquivo em `spiral_dataset/wavs/wavs/`)

Use `--spiral-audio-base` como o diretório **pai** dessas pastas (em geral a pasta `spiral_dataset` que contém `data.jsonl` e `wavs/`).

## Forma recomendada: `run_retrieval_eval.py --mode spiral`

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

Argumentos úteis:

| Flag | Descrição |
|------|-----------|
| `--spiral-audio-base` | Diretório base para resolver `spiral/wavs/...` |
| `--spiral-output-dir` | Pasta de saída (plots + JSON) |
| `--max-samples` | Limite de linhas para teste |
| `--batch-size-text` | Batch do `SentenceTransformer.encode` |
| `--hubert-model` | Default `facebook/hubert-large-ls960-ft` |
| `--sentence-transformer` | Default `sentence-transformers/LaBSE` |
| `--hits-k` | Ex.: `1,5,10,50` (igual ao modo matrix) |
| `--spiral-audio-pooling` | `mean` (default): média global dos chunks; `max_sim`: max-similaridade por chunk (estilo ColBERT) |
| `--spiral-chunk-samples` | Tamanho da janela em amostras 16 kHz (default `320000`); usado também para mapear `key_sentence_timestamp` → índice de chunk no modo `max_sim` |

## Avaliação com Docker

Os Dockerfiles ficam em `docker/` (`Dockerfile.amd64` para x86_64, `Dockerfile.arm64` para ARM64). Instalam o CLASP com `uv sync`. Para o eval SPIRAL são necessários **LaBSE** e **torchvision** (extras `realdata` ou `voxpopuli` no `pyproject.toml`). Construa a partir da **raiz do repo** (contexto `.`):

```bash
# x86_64 (troque por docker/Dockerfile.arm64 em máquina ARM64)
docker build -f docker/Dockerfile.amd64 -t clasp:spiral-eval \
  --build-arg UV_EXTRA_GROUPS=realdata \
  .
```

Se a sua equipe já builda outra tag (ex.: `clasp:arm64` / `clasp:amd64` do README principal) com o mesmo `UV_EXTRA_GROUPS`, use essa imagem no lugar de `clasp:spiral-eval`.

### `clasp:arm64` / `clasp:amd64` e erro `invalid choice: 'spiral'`

A imagem antiga só tinha `candidate` e `matrix`. **Rebuilde** com o código atual:

```bash
docker compose build clasp-arm64    # Apple Silicon / arm64
# em x86_64:
docker compose build clasp-amd64
```

O [`docker-compose.yml`](../docker-compose.yml) na raiz instala o extra **`voxpopuli`** (LaBSE, `torchvision`, etc.), compatível com o eval SPIRAL.

**Sem rebuild**, você pode montar `scripts` e `src` do host para forçar o código mais recente:

```text
-v "${REPO_ROOT}/scripts:/app/scripts" -v "${REPO_ROOT}/src:/app/src"
```

### Rodar o eval com o checkpoint do CLASP (GPU)

Monte o dataset, os pesos, a pasta de resultados e (recomendado) o cache do Hugging Face. Ajuste `REPO_ROOT` para o caminho absoluto do checkout.

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

- **Outro checkpoint**: altere `--model-path` para o `.pt` desejado (o arquivo deve existir em `models/` no host, montado em `/app/models`).
- **CPU**: remova `--gpus all` (fica muito mais lento).
- **Modelos do Hub** (HuBERT, LaBSE): na primeira execução podem ser baixados; o volume em `~/.cache/huggingface` evita re-downloads.

### Mesma pipeline via `run_spiral_retrieval.sh` no container

```bash
export REPO_ROOT="/caminho/absoluto/para/CLASP"

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

Omitir `MODEL_PATH` deixa o default do script (checkpoint em `models/checkpoints/`). Para outro `.pt`, use o caminho absoluto no container, p.ex. `-e MODEL_PATH=/app/models/checkpoints/outro_nome.pt`.

Argumentos adicionais após o script (ex. `--hits-k 1,5,10,50`) são repassados a `run_retrieval_eval.py`.

Detalhe dos WAVs e do `--spiral-audio-base` está em `spiral_dataset/wavs/README.md`.

## Pipeline automática (shell)

```bash
export SPIRAL_JSONL=/caminho/para/data.jsonl
export SPIRAL_AUDIO_BASE=/caminho/para/pasta_com_spiral
export MODEL_PATH=/caminho/para/CLASP.pt
export SPIRAL_OUTPUT_DIR=results/spiral
./scripts/run_spiral_retrieval.sh
```

Variáveis opcionais: `MAX_SAMPLES`, `HUBERT_MODEL`, `SENTENCE_TRANSFORMER`. Argumentos extras são repassados ao `run_retrieval_eval.py`.

## Wrapper legado

`scripts/eval_spiral_retrieval.py` apenas chama `clasp.evaluation.spiral_runner.run_spiral_retrieval_eval` (mesmo núcleo).

## Módulos Python

- `clasp.data.spiral`: `load_spiral_jsonl`, `spiral_temporal_bin_indices`
- `clasp.evaluation.spiral_runner`: orquestração embeddings + métricas + plots
- `clasp.evaluation.ranking_metrics`: `similarity_matrix_to_rows`, `compute_ranking_metrics`, `grouped_ranking_summary`
- `clasp.evaluation.retrieval_plots`: `save_retrieval_plot`, `save_grouped_hits_plot`

## Métricas

- Globais: `Hits@K` (taxa 0–1 no JSON; tabelas/console em %), `MRR`, `MAP`, `mean_rank`, `median_rank`.
- Por bin temporal: `Hits@1` por grupo (mesmo critério de rank 1-based que o restante do repo).

## Saídas (pasta `--spiral-output-dir`)

| Arquivo | Conteúdo |
|---------|----------|
| `retrieval_summary.png` | Barras Hits@K + histograma de ranks (modo percentual) |
| `retrieval_by_temporal_bin.png` | Hits@1 por bin temporal + contagem |
| `spiral_evaluation_results.json` | `ranking_metrics`, `grouped_temporal_bins`, metadados |

## Modo matrix: mesmo gráfico global

```bash
python scripts/run_retrieval_eval.py --mode matrix --dataset-path total.pkl \
  --retrieval-plot-dir results/matrix_plots --hits-k 1,5,10,50
```

Gera `retrieval_summary.png` nesse diretório, com a mesma função de plot usada no SPIRAL.

## Demo mock

`scripts/eval_spiral_retrieval_demo.py` permanece independente (dados sintéticos).

## Citação SPIRAL

Ver README do dataset em `spiral_dataset/README.md`.
