"""Mapear parágrafos SQuAD (Spoken-SQuAD) para ficheiros WAV em `dev_wav`.

Convenção dos ficheiros (índices alinhados com `enumerate` sobre `data` e `paragraphs`):
  ``{article_idx}_{paragraph_idx}_{sentence_idx}.wav``

O `sentence_idx` vem da geração do dataset e pode ter lacunas (ex.: 0,1,2,3,6,7).
Nunca assumir ``range(n)`` a partir do texto: usar apenas paths existentes ordenados
numericamente pelo terceiro componente.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from clasp.inference.audio_preprocess import load_mono_16k_padded


def paragraph_wav_paths(wav_dir: Path, article_idx: int, paragraph_idx: int) -> list[Path]:
    """Lista paths ``{article_idx}_{paragraph_idx}_*.wav`` ordenados por índice de frase."""
    prefix = f"{article_idx}_{paragraph_idx}_"
    numbered: list[tuple[int, Path]] = []
    pat = re.compile(rf"^{re.escape(str(article_idx))}_{re.escape(str(paragraph_idx))}_(\d+)\.wav$")
    for p in wav_dir.glob(f"{prefix}*.wav"):
        m = pat.match(p.name)
        if m:
            numbered.append((int(m.group(1)), p))
    numbered.sort(key=lambda x: x[0])
    return [p for _, p in numbered]


def concat_paragraph_wavs(paths: list[Path]) -> np.ndarray:
    """Mono 16 kHz, concatenação temporal (cada ficheiro via `load_mono_16k_padded`)."""
    if not paths:
        raise ValueError("concat_paragraph_wavs: empty paths")
    parts = [load_mono_16k_padded(p) for p in paths]
    return np.concatenate(parts, dtype=np.float32)


def iter_spoken_squad_paragraphs(squad_json: dict, wav_dir: Path) -> list[dict]:
    """Uma entrada por parágrafo: `article_idx`, `paragraph_idx`, `context`, `wav_paths`."""
    data = squad_json["data"]
    rows: list[dict] = []
    for article_idx, article in enumerate(data):
        for paragraph_idx, para in enumerate(article["paragraphs"]):
            paths = paragraph_wav_paths(wav_dir, article_idx, paragraph_idx)
            if not paths:
                raise FileNotFoundError(
                    f"Sem WAVs para artigo {article_idx} parágrafo {paragraph_idx} em {wav_dir}"
                )
            rows.append(
                {
                    "article_idx": article_idx,
                    "paragraph_idx": paragraph_idx,
                    "context": para["context"],
                    "wav_paths": paths,
                }
            )
    return rows
