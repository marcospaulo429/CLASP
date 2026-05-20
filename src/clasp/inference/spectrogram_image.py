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


def load_efficientnet_b7(device: torch.device) -> tuple[nn.Module, Callable]:
    weights = EfficientNet_B7_Weights.DEFAULT
    model = efficientnet_b7(weights=weights).to(device)
    model.eval()
    return model, weights.transforms()


def _spectrogram_pil(audio: np.ndarray) -> Image.Image:
    """STFT log spectrogram as RGB PIL image from a numpy audio array."""
    spec = librosa.stft(audio)
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


def spectrogram_pil_from_audio_path(audio_path: str) -> Image.Image:
    """STFT log spectrogram from a file path."""
    return _spectrogram_pil(load_mono_16k_padded(audio_path))


def spectrogram_pil_from_audio(audio: np.ndarray) -> Image.Image:
    """STFT log spectrogram from an in-memory audio array (e.g. after noise augmentation)."""
    return _spectrogram_pil(audio)


def _efficientnet_batch(
    items,
    get_audio_fn,
    vision_model: nn.Module,
    preprocess: Callable,
    device: torch.device,
    batch_size: int,
    desc: str,
) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    n = len(items)
    if n == 0:
        return out
    with torch.no_grad():
        with tqdm(total=n, desc=desc, unit="file") as pbar:
            for start in range(0, n, batch_size):
                chunk = items[start : start + batch_size]
                tensors = [preprocess(_spectrogram_pil(get_audio_fn(item))) for item in chunk]
                batch = torch.stack(tensors, dim=0).to(device)
                logits = vision_model(batch)
                for i in range(logits.size(0)):
                    out.append(logits[i].detach().cpu().float())
                pbar.update(len(chunk))
    return out


def efficientnet_embeddings_from_audio_paths(
    audio_paths: list[str],
    vision_model: nn.Module,
    preprocess: Callable,
    device: torch.device,
    batch_size: int = 4,
) -> list[torch.Tensor]:
    """Return one 1D float tensor [1000] per audio path."""
    return _efficientnet_batch(
        audio_paths, load_mono_16k_padded, vision_model, preprocess, device, batch_size,
        desc="spectrogram+EfficientNet",
    )


def efficientnet_embeddings_from_audio_arrays(
    audio_arrays: list[np.ndarray],
    vision_model: nn.Module,
    preprocess: Callable,
    device: torch.device,
    batch_size: int = 4,
) -> list[torch.Tensor]:
    """Return one 1D float tensor [1000] per in-memory audio array (e.g. after noise augmentation)."""
    return _efficientnet_batch(
        audio_arrays, lambda a: a, vision_model, preprocess, device, batch_size,
        desc="spectrogram+EfficientNet (noisy)",
    )
