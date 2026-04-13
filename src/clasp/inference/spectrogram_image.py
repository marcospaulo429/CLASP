"""Spectrogram rendering + EfficientNet-B7 image embeddings (see notebooks/clasp-inference.ipynb)."""

from __future__ import annotations

import io
from typing import Callable

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
) -> torch.Tensor:
    """Logits EfficientNet-B7 médios sobre janelas do sinal (áudio longo)."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        y = np.zeros(16_000, dtype=np.float32)
    chunks: list[np.ndarray] = []
    start = 0
    while start < y.size:
        end = min(start + chunk_samples, y.size)
        chunks.append(y[start:end])
        start = end
    logits_acc = None
    with torch.no_grad():
        for piece in chunks:
            if piece.size < 16_000:
                piece = np.pad(piece, (0, 16_000 - piece.size), mode="constant")
            pil = spectrogram_pil_from_waveform(piece)
            t = preprocess(pil).unsqueeze(0).to(device)
            logits = vision_model(t)[0].detach().cpu().float()
            logits_acc = logits if logits_acc is None else logits_acc + logits
    assert logits_acc is not None
    return logits_acc / len(chunks)


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
