# CLASP: Contrastive Language-Speech Pretraining for Multilingual Multimodal Information Retrieval

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2412.13071)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fmultimodalrag.github.io%2F)](https://clasp1.github.io/)

Original implementation of [CLASP: Contrastive Language-Speech Pretraining for Multilingual Multimodal Information Retrieval](https://arxiv.org/abs/2412.13071) (ECIR 2025).

[Models](https://huggingface.co/llm-lab/CLASP) | [SpeechBrown Dataset](https://huggingface.co/datasets/llm-lab/SpeechBrown) | [Springer](https://link.springer.com/chapter/10.1007/978-3-031-88717-8_2)

![CLASP-panel5](https://github.com/user-attachments/assets/472c5a52-29dd-4c59-af65-22a43fadc47e)

## Quick links

- **[Training guide](docs/TRAINING.md)** — how to train with VoxPopuli, Spoken SQuAD, and SPIRAL
- **[Evaluation guide](docs/EVAL.md)** — retrieval eval and noise robustness on each dataset

## Project structure

```text
.
├── src/clasp/          # library code (models, data, inference, evaluation)
├── scripts/            # train.py, build_*_pkl.py, run_*_eval.py
├── notebooks/          # original notebooks (unchanged)
├── models/checkpoints/ # *.pt weights
├── data/datasets/      # total_dataset_*.pkl
└── artifacts/          # embeddings, plots
```

See `src/clasp/` for the full module breakdown.

## Environment

```bash
uv sync                       # base deps
uv sync --extra realdata      # SpeechBrown / Spoken SQuAD
uv sync --extra voxpopuli     # VoxPopuli
```

Or with pip: `pip install -e .`

## Checkpoints

Download official weights from [llm-lab/CLASP](https://huggingface.co/llm-lab/CLASP):

```bash
huggingface-cli download llm-lab/CLASP CLASP_Concat_Final_Fusion_Encoder.pt \
  --local-dir models/checkpoints
```

For warm-start training use **`CLASP_Concat_Final_Fusion_Encoder.pt`** with `--init-checkpoint`.

## Abstract

CLASP is a multilingual, multimodal representation for audio-text information retrieval. It leverages the synergy between spoken content and textual data, combining audio spectrograms with HuBERT and a LaBSE sentence encoder. Evaluated across multiple languages, CLASP establishes new benchmarks in HITS@1, MRR, and meanR, outperforming ASR-based retrieval pipelines.

## Citation

```bibtex
@inproceedings{10.1007/978-3-031-88717-8_2,
  author = {Abootorabi, Mohammad Mahdi and Asgari, Ehsaneddin},
  title = {CLASP: Contrastive Language-Speech Pretraining for Multilingual Multimodal Information Retrieval},
  year = {2025},
  publisher = {Springer-Verlag},
  doi = {10.1007/978-3-031-88717-8_2},
  booktitle = {Advances in Information Retrieval: ECIR 2025, Lucca, Italy},
  pages = {10-20},
}
```

## Contact

- mahdi.abootorabi2@gmail.com
- asgari@berkeley.edu
