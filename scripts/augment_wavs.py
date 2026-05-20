#!/usr/bin/env python3
"""Generate noise-augmented WAV files from a directory of clean WAVs.

For each input WAV, writes one noisy copy per requested noise type:
    {stem}_white.wav, {stem}_reverb.wav, {stem}_ambient.wav

The output directory is a self-contained dataset that can be passed directly
to build_spoken_squad_pkl.py via --wav-dir.

Example
-------
# All three noise types (requires --esc50-dir for ambient):
python scripts/augment_wavs.py \\
    --wav-dir data/datasets/spoken_squad/train_wav \\
    --out-dir data/datasets/spoken_squad/train_wav_noisy \\
    --noise-types white reverb ambient \\
    --snr 20 \\
    --esc50-dir data/datasets/ESC-50 \\
    --copy-originals

# White + reverb only (no ESC-50 needed):
python scripts/augment_wavs.py \\
    --wav-dir data/datasets/spoken_squad/train_wav \\
    --out-dir data/datasets/spoken_squad/train_wav_noisy \\
    --noise-types white reverb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.audio.noise_augmentation import (
    add_ambient_noise,
    add_reverberation,
    add_white_noise,
    load_esc50_clip,
    scan_esc50_files,
)
from clasp.inference.audio_preprocess import load_mono_16k_padded

SAMPLE_RATE = 16_000


def augment_file(
    src: Path,
    out_dir: Path,
    noise_types: list[str],
    snr_db: float,
    esc50_files: list[Path] | None,
    overwrite: bool,
) -> int:
    """Write one noisy copy per noise type. Returns number of files written."""
    audio = load_mono_16k_padded(str(src))
    written = 0

    for noise_type in noise_types:
        dst = out_dir / f"{src.stem}_{noise_type}.wav"
        if dst.exists() and not overwrite:
            continue

        if noise_type == "white":
            noisy = add_white_noise(audio, snr_db=snr_db)
        elif noise_type == "reverb":
            noisy = add_reverberation(audio, sr=SAMPLE_RATE)
        elif noise_type == "ambient":
            noise_clip = load_esc50_clip(esc50_files, target_sr=SAMPLE_RATE)
            noisy = add_ambient_noise(audio, noise_clip, snr_db=snr_db)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        sf.write(str(dst), noisy, SAMPLE_RATE, subtype="PCM_16")
        written += 1

    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav-dir", type=Path, required=True,
                   help="Directory containing clean WAV files.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for noisy WAV files.")
    p.add_argument("--noise-types", nargs="+", default=["white", "reverb"],
                   choices=["white", "reverb", "ambient"],
                   help="Noise types to generate (default: white reverb).")
    p.add_argument("--snr", type=float, default=20.0,
                   help="SNR in dB for white/ambient noise (default: 20). Lower = more noise.")
    p.add_argument("--esc50-dir", type=Path, default=None,
                   help="ESC-50 dataset root (required when 'ambient' is in --noise-types).")
    p.add_argument("--copy-originals", action="store_true",
                   help="Also copy the clean originals into --out-dir.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-generate files that already exist in --out-dir.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    wav_files = sorted(args.wav_dir.glob("*.wav"))
    if not wav_files:
        raise SystemExit(f"No WAV files found in {args.wav_dir}")

    if "ambient" in args.noise_types:
        if args.esc50_dir is None:
            raise SystemExit("--esc50-dir is required when 'ambient' is in --noise-types.")
        esc50_files = scan_esc50_files(args.esc50_dir)
        print(f"Found {len(esc50_files)} ESC-50 clips in {args.esc50_dir}")
    else:
        esc50_files = None

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for src in tqdm(wav_files, desc="augmenting"):
        if args.copy_originals:
            dst = args.out_dir / src.name
            if not dst.exists() or args.overwrite:
                audio = load_mono_16k_padded(str(src))
                sf.write(str(dst), audio, SAMPLE_RATE, subtype="PCM_16")
                total_written += 1

        total_written += augment_file(
            src, args.out_dir, args.noise_types, args.snr, esc50_files, args.overwrite
        )

    n_input = len(wav_files)
    n_types = len(args.noise_types) + (1 if args.copy_originals else 0)
    print(
        f"Done. {n_input} input files × {n_types} variant(s) → "
        f"{total_written} files written to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
