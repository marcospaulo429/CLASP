# CLASP: Contrastive Language-Speech Pretraining for Multilingual Multimodal Information Retrieval

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2412.13071)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fmultimodalrag.github.io%2F)](https://clasp1.github.io/)

This repository includes the original implementation of
[CLASP: Contrastive Language-Speech Pretraining for Multilingual Multimodal Information Retrieval](https://arxiv.org/abs/2412.13071)
(published at [ECIR 2025](https://link.springer.com/chapter/10.1007/978-3-031-88717-8_2))
by Mohammad Mahdi Abootorabi and Ehsaneddin Asgari.

[Models](https://huggingface.co/llm-lab/CLASP) |
[Proposed Dataset](https://huggingface.co/datasets/llm-lab/SpeechBrown) |
[Springer Link](https://link.springer.com/chapter/10.1007/978-3-031-88717-8_2) |
[ACM Digital Library](https://dl.acm.org/doi/10.1007/978-3-031-88717-8_2)

## Notebook-Preserving Modularization

The notebook code has been extracted into a modular Python layout under `src/` and `scripts/`.
This refactor is intentionally **non-invasive**:

- Notebooks in `notebooks/` remain unchanged.
- No new algorithmic behavior was introduced.
- Training, inference, and retrieval/eval logic is organized into reusable modules.

## Project Structure

```text
.
├── pyproject.toml
├── uv.lock
├── notebooks/
│   ├── Training-Evaluation.ipynb
│   ├── clasp-inference.ipynb
│   └── ...
├── scripts/
│   ├── train.py
│   ├── run_inference.py
│   ├── run_retrieval_eval.py
│   ├── build_speechbrown_pkl.py
│   └── scan_speechbrown_audio.py
└── src/
    └── clasp/
        ├── config/
        │   └── settings.py
        ├── data/
        │   ├── datasets.py
        │   └── speechbrown_paths.py
        ├── models/
        │   └── fusion.py
        ├── train/
        │   └── trainer.py
        ├── inference/
        │   ├── audio_preprocess.py
        │   ├── embed_audio.py
        │   ├── spectrogram_image.py
        │   └── pipeline.py
        ├── retrieval/
        │   └── search.py
        └── evaluation/
            └── metrics.py
```

## Notebook -> Module Mapping

- `Training-Evaluation.ipynb`
  - model classes -> `src/clasp/models/fusion.py`
  - `CusDataset`, `TestDataset`, test metadata sampling -> `src/clasp/data/datasets.py`
  - `train_the_model` and training loop -> `src/clasp/train/trainer.py`
  - retrieval/evaluation metrics (`Hits@1`, `MRR`, `meanR`) -> `src/clasp/evaluation/metrics.py`
- `clasp-inference.ipynb`
  - HuBERT audio embedding extraction -> `src/clasp/inference/embed_audio.py`
  - spectrogram + EfficientNet -> `src/clasp/inference/spectrogram_image.py`
  - model loading and top-1 retrieval flow -> `src/clasp/inference/pipeline.py`
  - cosine ranking helpers -> `src/clasp/retrieval/search.py`

## Recommended data layout

Create these folders at the repository root (they are not committed by default if you add them to `.gitignore`). Large weights and datasets stay local.

```text
data/
  raw/                 # optional: original audio, metadata
  datasets/            # total_dataset_v*.pkl (aggregated splits)
models/
  checkpoints/         # *.pt — trained or downloaded CLASP weights
artifacts/
  embeddings/          # *.pt — stacked tensors or lists for run_inference.py
```

Typical CLI paths:

| Flag | Role | Example path |
|------|------|----------------|
| `--dataset-path` | Aggregated pickle | `data/datasets/total_dataset_v11.pkl` |
| `--save-path` | Output checkpoint after training | `models/checkpoints/my_clasp.pt` |
| `--model-path` | Load checkpoint (inference / eval candidate) | `models/checkpoints/clasp.pt` |
| `--audio-embeddings-path` | Audio-side vectors for inference | `artifacts/embeddings/audio.pt` |
| `--image-embeddings-path` | Image/spectrogram-side vectors | `artifacts/embeddings/image.pt` |
| `--text-embeddings-path` | Text encoder vectors | `artifacts/embeddings/text.pt` |

### Where to put checkpoints

| Command | Argument | What it is | Suggested folder |
|---------|----------|------------|------------------|
| `scripts/train.py` | `--save-path` | Trained model state | `models/checkpoints/` |
| `scripts/run_inference.py` | `--model-path` | CLASP `.pt` (trained or from [Hugging Face](https://huggingface.co/llm-lab/CLASP)) | `models/checkpoints/` |
| `scripts/run_retrieval_eval.py` | `--model-path` | Same (required for `--mode candidate` only) | `models/checkpoints/` |

Download official weights from [Models](https://huggingface.co/llm-lab/CLASP), place the file under `models/checkpoints/` (for example `models/checkpoints/clasp.pt`), and pass that path to `--model-path` or copy the path into `--save-path` when saving your own training run.

### Where to put embedding `.pt` files (inference only)

| Command | Arguments | What they are | Suggested folder |
|---------|-----------|---------------|------------------|
| `scripts/run_inference.py` | `--audio-embeddings-path`, `--image-embeddings-path`, `--text-embeddings-path` | PyTorch-saved tensors or lists of tensors, **aligned by row index** | `artifacts/embeddings/` |

`run_inference.py` loads each file with `torch.load` and optionally `torch.stack`s lists. You are responsible for producing the same alignment as in `clasp-inference.ipynb` (same ordering of samples).

## Environment with `uv`

Dependencies are declared in [`pyproject.toml`](pyproject.toml). Use [uv](https://docs.astral.sh/uv/) to create a reproducible environment.

1. Install uv (see the official installer for your OS).
2. From the repository root:

```bash
uv sync
```

This creates `.venv` and installs the locked dependencies (`uv.lock`).

3. Quick sanity check:

```bash
uv run python -m compileall src scripts
```

4. Run every script below with `uv run` so they use that environment (examples use the recommended paths).

Alternative without uv: `pip install -e .` or install the packages listed under `[project] dependencies` in `pyproject.toml`.

## SpeechBrown → total_dataset.pkl (real data)

Optional dependencies for building a pickle from [SpeechBrown](https://huggingface.co/datasets/llm-lab/SpeechBrown) metadata and local audio:

```bash
uv sync --extra realdata
```

**Prerequisite:** download SpeechBrown (for example `global_metadata.json` and `dataset_part1.zip` via the Hugging Face dataset page or CLI), unzip under your **dataset root** so the WAV files exist. Metadata often lists `dataset/part1/audios/...` while the zip unpacks to `dataset_part1/audios/...`; `build_speechbrown_pkl.py` tries both layouts.

Quick test build (caps rows before splitting):

```bash
uv run python scripts/build_speechbrown_pkl.py \
  --metadata-json path/to/global_metadata.json \
  --dataset-root path/to/folder_that_resolves_json_paths \
  --output data/datasets/total_dataset_real.pkl \
  --max-samples 120
```

**Audio quality:** HuBERT crashes on near-empty waveforms. Loading uses mono conversion, safe normalization, and pads short clips to 1s at 16 kHz (`audio_preprocess.py`). To **exclude** files shorter than a threshold before building (faster than padding edge cases), pass e.g. `--min-audio-seconds 0.5`. To **audit** bad rows: `uv run python scripts/scan_speechbrown_audio.py --metadata-json ... --dataset-root ... --min-audio-seconds 0.5 --output-bad bad_audio.json`.

**Note:** `run_retrieval_eval.py` defaults to `--num-candidates 100`. The test split must contain enough samples for that (roughly, test size should exceed `num-candidates`), or pass a smaller `--num-candidates` (for a tiny build, use at most `test_size - 1`).

Evaluate with the generated pickle and a CLASP checkpoint:

```bash
uv run python scripts/run_retrieval_eval.py \
  --dataset-path data/datasets/total_dataset_real.pkl \
  --mode candidate \
  --model-path models/checkpoints/clasp.pt \
  --audio-key hubert-emb \
  --text-key text \
  --threshold 0.5 \
  --num-candidates 10
```

Point `--model-path` at your CLASP `.pt` (for example a file downloaded from [Models](https://huggingface.co/llm-lab/CLASP)).

## End-to-end workflow

Follow these steps in order. Steps 1–2 depend on whether you already have the aggregated dataset and vectors.

### Step 1 — Aggregated dataset (`total_dataset_*.pkl`)

Training (`train.py`) and retrieval evaluation in **candidate** mode expect a pickle with top-level keys `train`, `validation`, and `test`. Each split is a dict that includes at least:

- `hubert-emb` (or `audio` if you configure `--audio-key` that way) — audio-side inputs to the fusion model
- `text` — text embeddings for contrastive training / queries
- `image` — image/spectrogram-side inputs to the fusion model

This matches the structure built in `Training-Evaluation.ipynb`. Place the file under `data/datasets/` (for example `data/datasets/total_dataset_v11.pkl`).

### Step 2 — Embeddings: two cases

**Case A — Vectors only inside the pickle (typical for train + candidate eval)**  
If `hubert-emb`, `text`, and `image` are already filled in the pickle, you **do not** need separate `.pt` files for `train.py` or `run_retrieval_eval.py --mode candidate`. Retrieval uses audio-derived representations through those keys.

**Case B — Separate `.pt` files for `run_inference.py`**  
`run_inference.py` does **not** read the big pickle; it needs three files:

1. Audio HuBERT-style embeddings (same logic as `hubert_audio_files` in `src/clasp/inference/embed_audio.py`).
2. Image/spectrogram embeddings (your notebook pipeline).
3. Text embeddings (e.g. sentence encoder).

Save them as `artifacts/embeddings/audio.pt`, `image.pt`, `text.pt` (or any names), then pass those paths to the three `--*-embeddings-path` flags. Indices must match across the three tensors.

To build audio vectors in code, load HuBERT with `transformers`, then call `hubert_audio_files(file_list, hubert_processor, hubert_model, device)` from `clasp.inference.embed_audio`, stack or save the list, and `torch.save` to disk. The reference notebook `notebooks/clasp-inference.ipynb` shows the full pipeline including fusion.

### Step 3 — Train (optional; or use a released checkpoint)

Writes the best checkpoint to `--save-path` (for example `models/checkpoints/my_model.pt`).

### Step 4 — Inference (optional)

Loads `--model-path` and the three embedding `.pt` files; prints top-1 match for `--sample-index`.

### Step 5 — Retrieval evaluation

- **Candidate mode** (`--mode candidate`): uses the test split inside the pickle, samples negative candidates, runs the loaded `--model-path`, reports metrics. No precomputed `clasp_emb` required.
- **Matrix mode** (`--mode matrix`): builds a full similarity matrix between `total_dataset['test'][emb-key]` and `total_dataset['test'][text-key]`. The default `--emb-key` is `clasp_emb`. That list must **already exist** on the test dict (filled by running the fusion model over the test split, as in `Training-Evaluation.ipynb`). If the key is missing or empty, matrix mode will fail.

### Matrix mode: `emb-key` and `clasp_emb`

For `--mode matrix`, `--emb-key` names the field on `total_dataset['test']` used as **query** embeddings (one row per test item). The script compares them to every column in `total_dataset['test'][text-key]`. Typical workflow from the paper/notebooks:

1. Load your trained checkpoint.
2. Run forward on the **test** split to append fused embeddings into `total_dataset['test']['clasp_emb']` (or another key you choose).
3. Save the updated pickle or keep it in memory, then run `run_retrieval_eval.py --mode matrix --emb-key clasp_emb` (or your key).

Per-source metrics add `--by-source` and use the `source` field on the test split.

## How to run (with `uv`)

All examples assume paths under the recommended layout. Adjust filenames as needed.

### 1) Training

Checkpoint output: **`--save-path`** → place under `models/checkpoints/`.

```bash
uv run python scripts/train.py \
  --dataset-path data/datasets/total_dataset_v11.pkl \
  --save-path models/checkpoints/my_clasp.pt \
  --audio-key hubert-emb \
  --text-key text \
  --num-epochs 100 \
  --learning-rate 5e-6
```

### 2) Inference (single text query index against paired embedding rows)

Checkpoints: **`--model-path`** → `models/checkpoints/`.  
Vectors: three **`--*-embeddings-path`** → `artifacts/embeddings/`.

```bash
uv run python scripts/run_inference.py \
  --model-path models/checkpoints/clasp.pt \
  --audio-embeddings-path artifacts/embeddings/audio.pt \
  --image-embeddings-path artifacts/embeddings/image.pt \
  --text-embeddings-path artifacts/embeddings/text.pt \
  --sample-index 0
```

### 3) Retrieval evaluation — candidate mode (notebook-style)

Requires **`--model-path`** under `models/checkpoints/` and **`--dataset-path`** under `data/datasets/`.

```bash
uv run python scripts/run_retrieval_eval.py \
  --dataset-path data/datasets/total_dataset_v11.pkl \
  --mode candidate \
  --model-path models/checkpoints/clasp.pt \
  --audio-key hubert-emb \
  --text-key text \
  --threshold 0.5 \
  --num-candidates 100
```

### 4) Retrieval evaluation — matrix mode

Requires `total_dataset['test'][emb-key]` to be populated **before** running (see “Matrix mode” above). Default `emb-key` is `clasp_emb`.

```bash
uv run python scripts/run_retrieval_eval.py \
  --dataset-path data/datasets/total_dataset_v11.pkl \
  --mode matrix \
  --emb-key clasp_emb \
  --text-key text \
  --threshold 0.5
```

Per-source metrics:

```bash
uv run python scripts/run_retrieval_eval.py \
  --dataset-path data/datasets/total_dataset_v11.pkl \
  --mode matrix \
  --emb-key clasp_emb \
  --text-key text \
  --threshold 0.5 \
  --by-source
```

## Notes

- Retrieval is performed on audio-derived vector representations (`hubert-emb` or `audio` keys when you set `--audio-key`, depending on your experiment).
- Matrix mode does not train the model; it only evaluates similarities using **precomputed** query embeddings under `--emb-key`.
- The modular code matches the notebook behavior; notebooks remain the baseline reference for exact preprocessing.

## Abstract

This study introduces CLASP (Contrastive Language-Speech Pretraining), a multilingual, multimodal representation tailored for audio-text information retrieval. CLASP leverages the synergy between spoken content and textual data. During training, we utilize our newly introduced speech-text dataset, which encompasses 15 diverse categories ranging from fiction to religion. CLASP's audio component integrates audio spectrograms with a pre-trained self-supervised speech model, while its language encoding counterpart employs a sentence encoder pre-trained on over 100 languages. This unified lightweight model bridges the gap between various modalities and languages, enhancing its effectiveness in handling and retrieving multilingual and multimodal data. Our evaluations across multiple languages demonstrate that CLASP establishes new benchmarks in HITS@1, MRR, and meanR metrics, outperforming traditional ASR-based retrieval methods that rely on transcribing speech into text for subsequent text retrieval in specific scenarios.

![CLASP-panel5](https://github.com/user-attachments/assets/472c5a52-29dd-4c59-af65-22a43fadc47e)

## Contributions

1. We introduce CLASP (Contrastive Language-Speech Pretraining), a novel lightweight multilingual, multimodal representation designed for audio-text retrieval.
2. We introduce a diverse paired speech-text dataset (Speech Brown) in 15 categories, encompassing a wide range of topics from fiction to religion.
3. We show that the combination of audio spectrograms with a pre-trained self-supervised speech model improves audio encoding in retrieval applications.
4. Evaluations in multiple languages demonstrate that CLASP sets new benchmarks in HITS@1, Mean Reciprocal Rank (MRR), and Mean Rank (meanR) metrics.

## Citation

If you find our paper, code, data, or models useful, please cite:

```bibtex
@inproceedings{10.1007/978-3-031-88717-8_2,
  author = {Abootorabi, Mohammad Mahdi and Asgari, Ehsaneddin},
  title = {CLASP: Contrastive Language-Speech Pretraining for Multilingual Multimodal Information Retrieval},
  year = {2025},
  isbn = {978-3-031-88716-1},
  publisher = {Springer-Verlag},
  address = {Berlin, Heidelberg},
  url = {https://doi.org/10.1007/978-3-031-88717-8_2},
  doi = {10.1007/978-3-031-88717-8_2},
  abstract = {This study introduces CLASP (Contrastive Language-Speech Pretraining), a multilingual, multimodal representation tailored for audio-text information retrieval. CLASP leverages the synergy between spoken content and textual data. During training, we utilize our newly introduced speech-text dataset, which encompasses 15 diverse categories ranging from fiction to religion. CLASP’s audio component integrates audio spectrograms with a pre-trained self-supervised speech model, while its language encoding counterpart employs a sentence encoder pre-trained on over 100 languages. This unified lightweight model bridges the gap between various modalities and languages, enhancing its effectiveness in handling and retrieving multilingual and multimodal data. Our evaluations across multiple languages demonstrate that CLASP establishes new benchmarks in HITS@1, MRR, and meanR metrics, outperforming traditional ASR-based retrieval methods that rely on transcribing speech into text for subsequent text retrieval, especially in specific scenarios.},
  booktitle = {Advances in Information Retrieval: 47th European Conference on Information Retrieval, ECIR 2025, Lucca, Italy, April 6-10, 2025, Proceedings, Part IV},
  pages = {10-20},
  numpages = {11},
  keywords = {Multimodal IR, Speech Retrieval, Contrastive Learning},
  location = {Lucca, Italy}
}
```

## Contact

If you have questions, please send an email to:
- mahdi.abootorabi2@gmail.com
- asgari@berkeley.edu
