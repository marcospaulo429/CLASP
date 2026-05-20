"""Noise augmentation for robustness evaluation.

Utilities to add white noise, ambient noise, and reverberation to audio samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.signal


def scan_esc50_files(esc50_dir: str | Path) -> list[Path]:
    """Return sorted list of WAV paths from an ESC-50 audio directory."""
    esc50_dir = Path(esc50_dir)
    candidates = [esc50_dir / "audio", esc50_dir]
    for d in candidates:
        files = sorted(d.glob("*.wav"))
        if files:
            return files
    raise FileNotFoundError(
        f"No WAV files found in {esc50_dir} or {esc50_dir / 'audio'}. "
        "Run scripts/download_esc50.sh first."
    )


def load_esc50_clip(esc50_files: list[Path], target_sr: int = 16000) -> np.ndarray:
    """Load a random ESC-50 clip and resample to target_sr (ESC-50 is 44100 Hz)."""
    import soundfile as sf

    wav_path = esc50_files[np.random.randint(len(esc50_files))]
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        n_samples = int(len(audio) * target_sr / sr)
        audio = scipy.signal.resample(audio, n_samples)
    return audio.astype(np.float32)


def add_white_noise(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add white Gaussian noise to audio at specified SNR.

    Args:
        audio: Input audio as float32 array.
        snr_db: Signal-to-noise ratio in dB (default 20 dB).
                Higher values = less noise, lower values = more noise.

    Returns:
        Audio with added noise, clipped to [-1, 1] range.
    """
    audio = np.asarray(audio, dtype=np.float32)

    # Compute signal power
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio

    # Compute noise power from SNR formula: SNR_dB = 10 * log10(P_signal / P_noise)
    noise_power = signal_power / (10 ** (snr_db / 10.0))

    # Generate and scale white noise
    noise = np.random.randn(len(audio)).astype(np.float32)
    noise_rms = np.sqrt(np.mean(noise ** 2))
    if noise_rms > 0:
        noise = noise * np.sqrt(noise_power) / noise_rms

    # Add noise and clip to valid range
    noisy_audio = audio + noise
    return np.clip(noisy_audio, -1.0, 1.0).astype(np.float32)


def add_ambient_noise(
    audio: np.ndarray, noise_audio: np.ndarray, snr_db: float = 20.0
) -> np.ndarray:
    """Add ambient noise (e.g., from WHAM) to audio at specified SNR.

    Args:
        audio: Input audio as float32 array.
        noise_audio: Ambient noise audio (should be same sample rate as audio).
        snr_db: Signal-to-noise ratio in dB.

    Returns:
        Audio with added ambient noise, clipped to [-1, 1] range.
    """
    audio = np.asarray(audio, dtype=np.float32)
    noise_audio = np.asarray(noise_audio, dtype=np.float32)

    # Tile or crop noise to match audio length
    if len(noise_audio) < len(audio):
        # Tile noise to match length
        n_repeats = (len(audio) // len(noise_audio)) + 1
        noise_audio = np.tile(noise_audio, n_repeats)[: len(audio)]
    else:
        # Crop noise to match length (random start position)
        start_idx = np.random.randint(0, len(noise_audio) - len(audio) + 1)
        noise_audio = noise_audio[start_idx : start_idx + len(audio)]

    # Compute power and scale noise by SNR
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio

    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise_rms = np.sqrt(np.mean(noise_audio ** 2))
    if noise_rms > 0:
        noise_audio = noise_audio * np.sqrt(noise_power) / noise_rms

    # Add noise and clip
    noisy_audio = audio + noise_audio
    return np.clip(noisy_audio, -1.0, 1.0).astype(np.float32)


def add_reverberation(audio: np.ndarray, decay_time_ms: float = 150.0, sr: int = 16000) -> np.ndarray:
    """Add synthetic reverberation using exponential decay impulse response.

    Args:
        audio: Input audio as float32 array.
        decay_time_ms: Time for impulse response to decay (in milliseconds, default 150 ms).
        sr: Sample rate (default 16000 Hz).

    Returns:
        Audio with added reverberation.
    """
    audio = np.asarray(audio, dtype=np.float32)

    decay_samples = max(2, int(decay_time_ms / 1000.0 * sr))
    t = np.arange(decay_samples, dtype=np.float32) / sr
    rir = np.exp(-3.0 * t / (decay_time_ms / 1000.0))
    rir[0] = 1.0
    early_idx = max(1, int(0.05 * sr))
    if early_idx < len(rir):
        rir[early_idx] += 0.5
    rir = rir / np.max(np.abs(rir))

    reverb_audio = scipy.signal.fftconvolve(audio, rir, mode="same")
    max_val = np.max(np.abs(reverb_audio))
    if max_val > 0:
        reverb_audio = reverb_audio / max_val * 0.95
    return np.clip(reverb_audio, -1.0, 1.0).astype(np.float32)


def add_reverberation_drr(
    audio: np.ndarray, drr_db: float = 0.0, decay_time_ms: float = 150.0, sr: int = 16000
) -> np.ndarray:
    """Add reverberation at a specified Direct-to-Reverberant Ratio (DRR).

    DRR = 10 * log10(P_direct / P_reverb).  Higher DRR = less reverb (cleaner).
    The paper (Tseng & Harwath, Interspeech 2025) sweeps DRR from +10 down to -20 dB.

    Args:
        audio: Input audio as float32 array.
        drr_db: Direct-to-Reverberant Ratio in dB (default 0 dB = equal direct/reverb energy).
        decay_time_ms: Controls RIR tail length (default 150 ms).
        sr: Sample rate (default 16000 Hz).

    Returns:
        Audio with reverberation matching the target DRR.
    """
    audio = np.asarray(audio, dtype=np.float32)

    decay_samples = max(2, int(decay_time_ms / 1000.0 * sr))
    t = np.arange(decay_samples, dtype=np.float32) / sr

    # Build reverb tail separately from the direct component
    reverb_tail = np.exp(-3.0 * t / (decay_time_ms / 1000.0)).astype(np.float32)
    reverb_tail[0] = 0.0  # direct component handled separately

    # 50 ms early reflection
    early_idx = max(1, int(0.05 * decay_time_ms / 1000.0 * sr))
    if early_idx < len(reverb_tail):
        reverb_tail[early_idx] += 0.5

    # Scale tail so that P_direct / P_reverb_tail = 10^(DRR/10)
    # P_direct = 1.0 (unit impulse), so P_reverb_target = 1 / 10^(DRR/10)
    p_reverb_current = float(np.sum(reverb_tail ** 2))
    if p_reverb_current > 0:
        p_reverb_target = 1.0 / (10.0 ** (drr_db / 10.0))
        reverb_tail = reverb_tail * float(np.sqrt(p_reverb_target / p_reverb_current))

    rir = np.zeros(decay_samples, dtype=np.float32)
    rir[0] = 1.0  # direct sound
    rir += reverb_tail

    reverb_audio = scipy.signal.fftconvolve(audio, rir, mode="same")
    max_val = np.max(np.abs(reverb_audio))
    if max_val > 0:
        reverb_audio = reverb_audio / max_val * 0.95
    return np.clip(reverb_audio, -1.0, 1.0).astype(np.float32)
