# Relatório de Mudanças — Tarefas SVQ/MSEB, RAG e Revisão de Docs

Resumo do que foi implementado no CLASP para (1) rodar o **benchmark MSEB** com o dataset
**SVQ** (reranking de hipóteses acústicas), (2) avaliar o **retrieval baseline do CLASP no SVQ
exatamente como nos outros datasets**, (3) avaliar um **pipeline RAG ponta-a-ponta**
(retrieval + geração), e (4) **reorganizar e traduzir a documentação**.

---

## Como o MSEB funciona (verificado a partir do código-fonte)

- **Base de encoder**: `mseb.encoder.MultiModalEncoder` — implementar `_setup()`,
  `_check_input_types(batch)`, `_encode(batch) -> Sequence[types.MultiModalEmbedding]`.
  `encode()`/`setup()` são `@final`. Um batch é uma `Sequence[types.MultiModalObject]` de
  `types.Sound` e/ou `types.Text`.
- **Reranking** (`mseb/evaluators/reranking_evaluator.py`): para cada `sound_id`, pontua o
  embedding de áudio contra cada embedding de texto candidato com `distance_fn=dot_product`,
  ordena, e reporta **MAP / MRR@10 / WER / CER** (texto top‑1 vs. verdade). Candidatos + ranks
  de referência vêm de `mseb/tasks/rerankings/query/svq.py` (nomes como `SVQEnUsQueryReranking`,
  26 locales).
- **Retrieval** (`mseb/evaluators/retrieval_evaluator.py`, tarefas em `mseb/tasks/retrievals/*/svq.py`:
  `document_in_lang`, `passage_in_lang`, `document_cross_lang`, …) — mesmo encoder, query de
  áudio vs. corpus de texto. **Requer `scann`** (só há wheels para x86‑Linux).
- **Driver**: `mseb/scripts/run_task.py` resolve um encoder por nome via `encoder_registry` e uma
  tarefa via `tasks.get_task_by_name(name)`, roda um `runner.DirectRunner`, depois
  `leaderboard.run_benchmark`. Evitamos alterar o registry entregando **nosso próprio driver
  fino** que instancia o encoder do CLASP diretamente.

Isso mapeia no CLASP **sem mudanças no modelo**: Sound → embedding de áudio fundido do CLASP
768‑d; Text → LaBSE 768‑d; ambos **L2‑normalizados**, de forma que `dot_product` == o cosseno
com que o CLASP foi treinado.

### Restrições de ambiente descobertas (importantes)

- **O MSEB exige Python ≥ 3.12**; o venv base do CLASP é 3.10. Por isso usamos um ambiente
  dedicado (`.venv-mseb`, criado com `uv venv --python 3.12`). O CLASP também suporta 3.12,
  então convivem nesse ambiente.
- O avaliador de reranking importa `whisper.normalizers`, que precisa do **`openai-whisper`**
  (o pacote `whisper` do PyPI é outro projeto e não tem esse módulo) — fixado no extra `[mseb]`.
- No **Apple Silicon (ARM macOS)** o carregador do SVQ (`array_record` / `apache_beam`)
  **dá segfault** ao ser carregado junto com o torch; portanto a execução completa das tarefas
  acontece em **Linux/x86 ou Docker** (o repo já tem Dockerfiles). Os *módulos* de reranking e o
  avaliador importam normalmente no mac; só a I/O do dataset é bloqueada.
- **`mseb` não está no PyPI** — o extra usa dependência via git.

---

## Workstream A — Adapter CLASP↔MSEB (reranking)

Novo módulo **`src/clasp/mseb_adapter/clasp_encoder.py`** — `ClaspMultiModalEncoder(MultiModalEncoder)`:
- `_setup()`: carrega o checkpoint de fusão do CLASP e os backbones congelados reutilizando
  utilitários existentes — `clasp.inference.pipeline.load_model`,
  `clasp.inference.embed_audio.hubert_numpy_waveform` (HuBERT `facebook/hubert-large-ls960-ft` via
  `transformers.AutoProcessor`/`HubertModel`), `clasp.inference.spectrogram_image.load_efficientnet_b7`
  + `efficientnet_embedding_from_waveform`, e LaBSE via
  `sentence_transformers.SentenceTransformer("sentence-transformers/LaBSE")`.
- `_check_input_types(batch)`: aceita `types.Sound` **e** `types.Text` (reranking mistura os dois).
- `_encode(batch)`: por objeto —
  - `types.Sound` → `encoder.resample_sound(..., 16000)` → HuBERT(1024) + EfficientNet(1000) →
    `load_model` forward → 768‑d → L2‑normaliza → `types.SoundEmbedding` (shape `[1, 768]`).
  - `types.Text` → LaBSE encode → 768‑d → L2‑normaliza → `types.TextEmbedding`.

Novo driver **`scripts/run_mseb_task.py`** (espelha `mseb/scripts/run_task.py`, mas instancia
nosso encoder diretamente, sem alterar o registry): args `--task`, `--model-path`,
`--results-jsonl`, `--batch-size`, `--device`, `--chunk-batch-size`, `--task-module`. Importa
**apenas o módulo da tarefa necessária** (evita o import pesado de *todas* as tarefas do MSEB, que
é instável em algumas plataformas), monta `DirectRunner(encoder=...)`,
`get_task_by_name(--task)`, `task.setup(runner=...)`, `leaderboard.run_benchmark(...)`, e grava
JSONL. Dirige tanto reranking (`SVQEnUsQueryReranking`, …) quanto retrieval do SVQ
(`SVQEnUsDocumentInLangRetrieval`, …) no Linux/x86.

Dependências: extra em **`pyproject.toml`** → `[mseb]` (`mseb @ git+…`, `openai-whisper`,
`sentence-transformers`, `torchvision`, `Pillow`). Isolado da instalação base (Python 3.12).

Wrapper de experimentos **`experiments/run_svq_mseb.sh`** (estilo de `experiments/eval_spoken_squad.sh`):
recebe `<MODEL> [TASKS...]`, grava logs/JSONL em `logs/` + `artifacts_user/`.

---

## Workstream B — Retrieval baseline do SVQ (**exatamente igual aos outros datasets**)

Objetivo do usuário: comparar o **CLASP baseline (sem adições)** no SVQ usando o **mesmo pipeline**
dos outros datasets (VoxPopuli / Spoken‑SQuAD), para ver se o novo dado rende melhor ou pior.

Novo **`scripts/build_svq_pkl.py`** (espelha `build_voxpopuli_pkl.py`): carrega o `google/svq`
e pareia cada query falada com **sua própria transcrição** (campo `text` do SVQ) — ou seja,
auto‑retrieval do CLASP (áudio → texto falado), idêntico ao pareamento áudio↔`normalized_text`
do VoxPopuli. Saída por split: `hubert-emb`[1024], `text`[768], `image`[1000], `audio_path` —
**mesmo schema** dos PKLs existentes. Config padrão: `audio_en_us_clean` (~1.4k linhas de inglês
limpo). Usa `--config audio --locale <loc>` (+ `--max-samples`) para outros locales.

O PKL roda pelo **`scripts/run_retrieval_eval.py` sem nenhuma modificação**, produzindo o mesmo
dicionário de métricas (Hits@1, MRR, etc.) — comparação apples‑to‑apples com os outros datasets.

Wrapper **`experiments/run_svq_retrieval.sh`**: `<MODEL> [CONFIG] [MAX_SAMPLES]` → constrói o PKL
+ roda o retrieval eval sobre o split inteiro.

> **Nota:** "validação de retrieval" aqui = **somente métricas de retrieval** (o modelo não gera
> nada). Isso já é exatamente o que o `run_retrieval_eval.py` faz.

---

## Workstream C — RAG ponta‑a‑ponta no SVQ (harness nativo)

Objetivo do usuário: avaliar o **pipeline RAG completo** (retrieval **+ geração**), não só retrieval.
O SVQ fornece: query de áudio + passagem de referência + span de resposta de referência, então dá
para montar um RAG de QA falado.

Harness nativo (independente do pacote `mseb`, roda no **venv base 3.10** e numa GPU CUDA como a 4090):

- **`src/clasp/rag/svq_rag.py`** — `ClaspEmbedder` (áudio via CLASP; passagens via LaBSE; mesmo
  espaço 768‑d), carregador `load_svq_reasoning_rows` (junta a config de raciocínio ao áudio pelo
  `utt_id` — as configs de reasoning têm `utt_id` mas **não** têm o waveform; join **em streaming**
  a partir da config `audio`, sem download de vários GB), `build_corpus` e `run_svq_rag`.
- **`src/clasp/rag/generator.py`** — `HFGenerator` (LLM local, padrão **Qwen3‑8B**, configurável;
  Qwen3 com *thinking* desligado por padrão) e `DryRunGenerator` (stub sem pesos, para testar o
  pipeline sem LLM).
- **`src/clasp/evaluation/qa_metrics.py`** — EM / F1 estilo SQuAD (normalização unicode‑safe,
  tratamento de "sem resposta"/unanswerable).
- **`scripts/run_svq_rag_eval.py`** — CLI fina: retrieval CLASP (top‑k) → geração → EM/F1 vs.
  spans de referência, reportando também **Recall@k** (retrieval e geração ficam separáveis).
- Wrapper **`experiments/run_svq_rag.sh`** (`DRY_RUN=1`, `GENERATOR=<hf-id>` como overrides).

O texto da pergunta passado ao LLM é a **transcrição de referência do SVQ**, então a métrica
isola retrieval do CLASP + geração do LLM (sem erro de ASR — o usuário descartou a validação de ASR).

Dependências: extra **`[rag]`** no `pyproject.toml` (`datasets`, `sentence-transformers`,
`torchvision`, `Pillow`, `transformers>=4.51` para o Qwen3, `accelerate`). Roda no venv base 3.10.

---

## Workstream D — Docs: reorganizar + traduzir (EN) + tirar defasagem

- **Traduzido PT→EN**: `docs/EVAL.md`, `docs/TRAINING.md`, `scripts/SPIRAL_EVAL_README.md`.
- **Corrigida a defasagem** de `docs/EVAL.md` para `run_noise_robustness_eval.py` (o argparse é a
  fonte da verdade): removidos `--audio-paths-from-pickle`, `--wav-dir`, `--train-json`,
  `--num-candidates`; documentado o comportamento atual — autodetecção de `audio_paths`/`audio_path`
  e modo de retrieval derivado de `_meta.pooling_mode`, além de `--noise-types`,
  `--esc50-dir`/`--wham-dir`, `--ambient-num-samples`, `--snr-levels`, `--chunk-batch-size`.
- **Realocado** o roadmap de pesquisa de `src/clasp/data/testes.md` (mal‑posto num diretório de
  código) → **`docs/ROADMAP.md`** (traduzido).
- **Novo `docs/MSEB.md`**: como instalar o extra `[mseb]` (Python 3.12) e rodar reranking +
  retrieval do SVQ via `scripts/run_mseb_task.py` / `experiments/run_svq_mseb.sh`, com a lista de
  nomes de tarefas e métricas; e uma seção de **RAG ponta‑a‑ponta**.
- **`docs/EVAL.md`** ganhou a seção **§4 SVQ (retrieval baseline)**.
- **`README.md`**: estrutura do projeto + Quick links atualizados (MSEB, ROADMAP, módulo `rag/`).
- **`.gitignore`**: adicionados `.venv/` e `.venv-mseb/`.

---

## Arquivos críticos

- **Novos**:
  - MSEB: `src/clasp/mseb_adapter/{__init__.py,clasp_encoder.py}`, `scripts/run_mseb_task.py`,
    `experiments/run_svq_mseb.sh`.
  - Retrieval baseline SVQ: `scripts/build_svq_pkl.py`, `experiments/run_svq_retrieval.sh`.
  - RAG: `src/clasp/rag/{__init__.py,svq_rag.py,generator.py}`,
    `src/clasp/evaluation/qa_metrics.py`, `scripts/run_svq_rag_eval.py`,
    `experiments/run_svq_rag.sh`.
  - Docs: `docs/MSEB.md`, `docs/ROADMAP.md`, este `RELATORIO_SVQ_MSEB.md`.
- **Reutilizados (sem alteração)**: `src/clasp/inference/pipeline.py` (`load_model`),
  `src/clasp/inference/embed_audio.py` (`hubert_numpy_waveform`, `hubert_audio_files`),
  `src/clasp/inference/spectrogram_image.py` (`load_efficientnet_b7`,
  `efficientnet_embedding_from_waveform`, `efficientnet_embeddings_from_audio_paths`),
  `src/clasp/models/fusion.py`, `scripts/run_retrieval_eval.py` (o retrieval do SVQ passa por ele
  **sem modificação**).
- **Editados**: `pyproject.toml` (extras `[mseb]` e `[rag]`), `docs/EVAL.md`, `docs/TRAINING.md`,
  `scripts/SPIRAL_EVAL_README.md`, `README.md`, `.gitignore`; **removido**
  `src/clasp/data/testes.md` (movido para `docs/ROADMAP.md`).

---

## Verificação

O que foi **rodado e passou localmente** (Apple Silicon), e o que fica para o box Linux/4090:

1. **Adapter MSEB (verificado)**: com modelos reais do CLASP (HuBERT/EfficientNet/LaBSE/checkpoint
   de fusão) contra o **`RerankingEvaluator` real do MSEB** (entradas sintéticas) → embeddings
   `[1,768]` unit‑norm corretos e MAP/MRR/WER/CER dentro do intervalo. Ambiente `.venv-mseb` (3.12)
   com `mseb` + `openai-whisper` importando OK.
2. **Retrieval baseline SVQ (verificado ponta‑a‑ponta)**: build de 3 linhas do `audio_en_us_clean`
   → PKL com schema idêntico (`hubert-emb`[1024]/`text`[768]/`image`[1000]/`audio_path`) → o
   **`run_retrieval_eval.py` sem modificação** rodou e emitiu o mesmo dicionário (Hits@1, MRR, …).
3. **RAG (núcleo verificado)**: `qa_metrics` (EM/F1 + unanswerable) e `run_svq_rag` com retrieval
   real do CLASP + `--dry-run-generator` → Recall@k, EM e F1 computados e no intervalo. A chamada
   real ao **Qwen3‑8B** fica para a GPU (código transformers padrão).
4. **Isolamento**: `import clasp` e os scripts de retrieval continuam funcionando no venv base 3.10
   (o `mseb_adapter` só importa `mseb` quando usado; o `rag/` importa `datasets`/`transformers`
   de forma preguiçosa).

### Como rodar no box Linux/4090

```bash
# baseline de retrieval (venv base)
uv pip install -e ".[rag]"
bash experiments/run_svq_retrieval.sh models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt

# RAG ponta-a-ponta (Qwen3-8B)
bash experiments/run_svq_rag.sh models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt en_us

# reranking MSEB (ambiente 3.12 dedicado — instale o mseb isolado p/ evitar o
# backtrack do apache-beam 2.3.0 que quebra no 3.12)
uv venv --python 3.12 .venv-mseb
VIRTUAL_ENV=.venv-mseb uv pip install "mseb @ git+https://github.com/google-research/mseb.git" openai-whisper
VIRTUAL_ENV=.venv-mseb uv pip install sentence-transformers torchvision matplotlib
VIRTUAL_ENV=.venv-mseb uv pip install -e . --no-deps
.venv-mseb/bin/python scripts/run_mseb_task.py \
  --task SVQEnUsQueryReranking \
  --model-path models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt \
  --device cuda \
  --results-jsonl artifacts_user/svq_rerank_enus.jsonl
```

### Pendências / observações

- **Reranking/Retrieval MSEB completos**: ainda não rodados de ponta a ponta (bloqueados no mac por
  segfault do `array_record` e por `scann`); rodar no Linux/x86 ou Docker. Comparar contra baselines
  em `mseb/results/*` e o [leaderboard](https://google-research.github.io/mseb/leaderboard.html).
- **RAG multi‑locale em escala**: o join `--audio-config audio` faz streaming do conjunto de áudio
  completo para achar os `utt_id`; ok para `--max-samples` moderado, mas para varreduras grandes
  vale pré‑indexar `utt_id`→shard uma vez.
- **Bug pré‑existente (fora do escopo, não corrigido)**: `clasp.evaluation.spiral_runner` importa
  `clasp.data.spiral`, mas `src/clasp/data/spiral.py` não existe — o modo SPIRAL pode estar quebrado.
