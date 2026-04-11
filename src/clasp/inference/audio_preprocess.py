"""Load mono 16 kHz audio with padding so HuBERT / spectrogram pipelines never see empty inputs."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16_000
# HuBERT CNN stack needs enough samples; 1s at 16 kHz is safe.
MIN_SAMPLES_16K = 16_000


def load_mono_16k_padded(file_path: str | Path) -> np.ndarray:
    """Read WAV (or format supported by soundfile), mono, 16 kHz, peak-normalize, pad to MIN if short."""
    data, samplerate = sf.read(str(file_path), always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = np.asarray(data, dtype=np.float32)
    peak = float(np.max(np.abs(data)))
    if peak > 0:
        data = data / peak
    data = librosa.resample(data, orig_sr=samplerate, target_sr=TARGET_SR)
    if len(data) < MIN_SAMPLES_16K:
        data = np.pad(data, (0, MIN_SAMPLES_16K - len(data)), mode="constant")
    return data


def audio_duration_seconds(file_path: str | Path) -> float:
    """Duration in seconds (from header; does not decode full file)."""
    info = sf.info(str(file_path))
    return info.frames / float(info.samplerate)
