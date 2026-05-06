"""Spectrogram rendering + EfficientNet-B7 image embeddings (see notebooks/clasp-inference.ipynb)."""

from __future__ import annotations

import io
from typing import Callable, Literal

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch import nn
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7

from clasp.inference.audio_preprocess import load_mono_16k_padded

Pooling = Literal["mean", "multivector"]


def spectrogram_pil_from_waveform(y: np.ndarray) -> Image.Image:
    """Igual a `spectrogram_pil_from_audio_path`, mas a partir de mono 16 kHz já carregado."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    spec = librosa.stft(y)
    spec_db = librosa.amplitude_to_db(np.abs(spec))

    fig, ax = plt.subplots(figsize=(4, 4))
    librosa.display.specshow(spec_db, x_axis="time", y_axis="log", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def efficientnet_embedding_from_waveform(
    y: np.ndarray,
    vision_model: nn.Module,
    preprocess: Callable,
    device: torch.device,
    chunk_samples: int = 320_000,
    chunk_batch_size: int = 1,
    *,
    pooling: Pooling = "mean",
) -> torch.Tensor:
    """EfficientNet-B7 classifier logits on fixed windows (long audio).

    * ``mean``: one [1000] vector per file (mean over window logits).
    * ``multivector``: ``[num_chunks, 1000]`` aligned with HuBERT chunking.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        y = np.zeros(16_000, dtype=np.float32)
    chunks: list[np.ndarray] = []
    start = 0
    while start < y.size:
        end = min(start + chunk_samples, y.size)
        chunks.append(y[start:end])
        start = end
    all_logits: list[torch.Tensor] = []
    bs = max(1, int(chunk_batch_size))
    with torch.no_grad():
        for i in range(0, len(chunks), bs):
            batch_pieces = chunks[i : i + bs]
            tensors: list[torch.Tensor] = []
            for piece in batch_pieces:
                if piece.size < 16_000:
                    piece = np.pad(piece, (0, 16_000 - piece.size), mode="constant")
                pil = spectrogram_pil_from_waveform(piece)
                tensors.append(preprocess(pil))
            batch_t = torch.stack(tensors).to(device)
            logits = vision_model(batch_t).detach().cpu().float()
            for row in range(logits.size(0)):
                all_logits.append(logits[row].clone())
    if not all_logits:
        return torch.zeros(1000, dtype=torch.float32) if pooling == "mean" else torch.zeros(1, 1000, dtype=torch.float32)
    stacked = torch.stack(all_logits, dim=0)
    if pooling == "multivector":
        return stacked
    if pooling == "mean":
        return torch.mean(stacked, dim=0)
    raise ValueError(f"pooling must be 'mean' or 'multivector', got {pooling!r}")


def load_efficientnet_b7(device: torch.device) -> tuple[nn.Module, Callable]:
    weights = EfficientNet_B7_Weights.DEFAULT
    model = efficientnet_b7(weights=weights).to(device)
    model.eval()
    return model, weights.transforms()


def spectrogram_pil_from_audio_path(audio_path: str) -> Image.Image:
    """STFT log spectrogram as RGB PIL image (notebook-style librosa.display.specshow)."""
    y = load_mono_16k_padded(audio_path)
    spec = librosa.stft(y)
    spec_db = librosa.amplitude_to_db(np.abs(spec))

    fig, ax = plt.subplots(figsize=(4, 4))
    librosa.display.specshow(spec_db, x_axis="time", y_axis="log", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def efficientnet_embeddings_from_audio_paths(
    audio_paths: list[str],
    vision_model: nn.Module,
    preprocess: Callable,
    device: torch.device,
    batch_size: int = 4,
) -> list[torch.Tensor]:
    """Return one 1D float tensor [1000] per audio path (classifier logits as in the reference notebook)."""
    out: list[torch.Tensor] = []
    n = len(audio_paths)
    if n == 0:
        return out
    with torch.no_grad():
        with tqdm(total=n, desc="spectrogram+EfficientNet", unit="file") as pbar:
            for start in range(0, n, batch_size):
                chunk = audio_paths[start : start + batch_size]
                tensors = []
                for p in chunk:
                    pil = spectrogram_pil_from_audio_path(p)
                    tensors.append(preprocess(pil))
                batch = torch.stack(tensors, dim=0).to(device)
                logits = vision_model(batch)
                for i in range(logits.size(0)):
                    out.append(logits[i].detach().cpu().float())
                pbar.update(len(chunk))
    return out
