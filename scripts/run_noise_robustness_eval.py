#!/usr/bin/env python3
"""Noise robustness evaluation: measure retrieval performance under noise.

Methodology follows Tseng & Harwath (Interspeech 2025):
  - White noise  : swept by SNR (dB) — Gaussian noise at specified signal-to-noise ratio
  - Ambient noise: swept by SNR (dB) — WHAM! real-world background sounds
  - Reverberation: swept by DRR (dB) — Direct-to-Reverberant Ratio via synthetic RIR

Default SNR range : 30, 20, 10, 5, 0, -5, -10 dB  (paper: -10 to 30)
Default DRR range : 10, 5, 0, -5, -10, -15, -20 dB (paper: -20 to 10)

Vision embeddings (EfficientNet spectrogram) are computed from the NOISY audio
to keep the evaluation consistent across both encoder branches.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoProcessor, HubertModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.audio.noise_augmentation import (
    add_ambient_noise,
    add_reverberation_drr,
    add_white_noise,
    load_esc50_clip,
    scan_esc50_files,
)
from clasp.config.settings import get_default_device
from clasp.data.datasets import TestDataset, build_test_metadata
from clasp.evaluation.metrics import evaluate_model_on_candidates
from clasp.inference.audio_preprocess import TARGET_SR, load_mono_16k_padded
from clasp.inference.pipeline import load_model
from clasp.inference.spectrogram_image import (
    efficientnet_embeddings_from_audio_arrays,
    load_efficientnet_b7,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-path", required=True, help="Path to total_dataset_spoken_squad.pkl")
    p.add_argument("--model-path", required=True, help="Path to CLASP checkpoint")
    p.add_argument(
        "--train-json",
        type=Path,
        default=Path("data/datasets/spoken_squad/spoken_train-v1.1.json"),
        help="Spoken SQuAD JSON to reconstruct audio paths",
    )
    p.add_argument(
        "--wav-dir",
        type=Path,
        default=Path("data/datasets/spoken_squad/train_wav"),
        help="Directory with WAV files",
    )
    p.add_argument("--audio-key", default="hubert-emb")
    p.add_argument("--text-key", default="text")
    p.add_argument("--num-candidates", type=int, default=10)
    p.add_argument(
        "--snr-levels",
        default="30,20,10,5,0,-5,-10",
        help="SNR levels in dB for white/ambient noise (paper: 30,20,10,5,0,-5,-10)",
    )
    p.add_argument(
        "--drr-levels",
        default="10,5,0,-5,-10,-15,-20",
        help="DRR levels in dB for reverberation (paper: 10,5,0,-5,-10,-15,-20). "
             "Higher DRR = less reverb.",
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--ambient-dir",
        default=None,
        help="Directory with ambient noise WAVs (WHAM! or ESC-50). "
             "If None, ambient noise eval is skipped.",
    )
    p.add_argument(
        "--ambient-source",
        default="esc50",
        choices=["esc50", "wham"],
        help="Format of --ambient-dir (default: esc50).",
    )
    p.add_argument("--output-csv", default=None, help="Path to save results as CSV")
    p.add_argument("--hubert-model", default="facebook/hubert-large-ls960-ft")
    p.add_argument("--vision-batch-size", type=int, default=4)
    p.add_argument("--max-test-samples", type=int, default=None)
    # W&B
    p.add_argument("--wandb-project", default=None, help="W&B project name. Omit to disable.")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_audio_paths(json_path: Path, wav_dir: Path, limit: int | None) -> list[str]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    paths: list[str] = []
    for a_idx, article in enumerate(data["data"]):
        for p_idx, para in enumerate(article["paragraphs"]):
            for q_idx in range(len(para["qas"])):
                wav = wav_dir / f"{a_idx}_{p_idx}_{q_idx}.wav"
                if wav.is_file():
                    paths.append(str(wav))
                    if limit is not None and len(paths) >= limit:
                        return paths
    return paths


def _load_ambient_files(ambient_dir: Path, source: str) -> list[Path]:
    if source == "esc50":
        return scan_esc50_files(ambient_dir)
    # WHAM: look for *.wav anywhere under ambient_dir
    files = sorted(ambient_dir.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files found under {ambient_dir}")
    return files


def _load_random_ambient(ambient_files: list[Path]) -> np.ndarray:
    from clasp.audio.noise_augmentation import load_esc50_clip
    return load_esc50_clip(ambient_files, target_sr=TARGET_SR)


def _apply_noise(
    audio: np.ndarray,
    noise_type: str,
    level: float,
    ambient_files: list[Path] | None,
) -> np.ndarray:
    """Apply one noise type at the given level.

    level = SNR (dB) for white/ambient; DRR (dB) for reverb.
    """
    if noise_type == "white":
        return add_white_noise(audio, snr_db=level)
    if noise_type == "ambient":
        noise_clip = _load_random_ambient(ambient_files)  # type: ignore[arg-type]
        return add_ambient_noise(audio, noise_clip, snr_db=level)
    if noise_type == "reverb":
        return add_reverberation_drr(audio, drr_db=level)
    raise ValueError(f"Unknown noise type: {noise_type}")


def _embed_noisy(
    audio_paths: list[str],
    noise_type: str,
    level: float,
    hubert_processor,
    hubert_model: HubertModel,
    vision_model,
    vision_preprocess,
    device: torch.device,
    vision_batch_size: int,
    ambient_files: list[Path] | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Re-embed audio with noise applied to BOTH HuBERT and vision branches."""
    hubert_embeddings: list[torch.Tensor] = []
    noisy_arrays: list[np.ndarray] = []

    for path in audio_paths:
        audio = load_mono_16k_padded(path)
        noisy = _apply_noise(audio, noise_type, level, ambient_files)
        noisy_arrays.append(noisy)

        t = torch.from_numpy(noisy)
        inputs = hubert_processor(t, sampling_rate=TARGET_SR, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = hubert_model(**inputs).last_hidden_state
            hubert_embeddings.append(torch.mean(hidden, dim=1).squeeze(0).cpu())

    # Vision branch also uses the noisy audio (not clean paths)
    vision_embeddings = efficientnet_embeddings_from_audio_arrays(
        noisy_arrays, vision_model, vision_preprocess, device, batch_size=vision_batch_size
    )
    return hubert_embeddings, vision_embeddings


def _run_condition(
    label: str,
    noise_type: str,
    level: float,
    level_axis: str,
    audio_paths: list[str],
    hubert_processor,
    hubert_model,
    vision_model,
    vision_preprocess,
    device: torch.device,
    vision_batch_size: int,
    ambient_files: list[Path] | None,
    test_data: dict,
    test_metadata: list,
    audio_key: str,
    text_key: str,
    clasp_model,
    all_results: dict,
    results_table,
    metric_keys: list[str],
    use_wandb: bool,
) -> None:
    print(f"\nEvaluating: {label}")
    hubert_embs, vision_embs = _embed_noisy(
        audio_paths, noise_type, level,
        hubert_processor, hubert_model, vision_model, vision_preprocess,
        device, vision_batch_size, ambient_files,
    )
    noisy_data = {
        audio_key: hubert_embs,
        "image": vision_embs,
        text_key: test_data[text_key],
    }
    loader = DataLoader(
        TestDataset(noisy_data, test_metadata, audio_key, text_key),
        batch_size=1, shuffle=False,
    )
    metrics = evaluate_model_on_candidates(clasp_model, loader, device, threshold=0.5)
    all_results[label] = metrics
    print(f"  Hits@1: {metrics['Hits@1']:.4f}  MRR: {metrics['MRR']:.4f}")

    if use_wandb:
        wandb.log({
            f"{noise_type}/Hits@1": metrics["Hits@1"],
            f"{noise_type}/MRR": metrics["MRR"],
            f"{noise_type}/Macro_F1": metrics["Macro F1"],
            f"{noise_type}/Accuracy": metrics["Accuracy"],
            f"{noise_type}/Golden_Accuracy": metrics["Golden Accuracy"],
            level_axis: level,
        })
        if results_table is not None:
            results_table.add_data(label, noise_type, level_axis, level,
                                   *[metrics[k] for k in metric_keys])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()

    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "model_path": str(args.model_path),
                "dataset_path": str(args.dataset_path),
                "hubert_model": args.hubert_model,
                "num_candidates": args.num_candidates,
                "snr_levels": args.snr_levels,
                "drr_levels": args.drr_levels,
                "max_test_samples": args.max_test_samples,
                "ambient_source": args.ambient_source,
            },
        )

    # --- Load data ---
    with open(args.dataset_path, "rb") as f:
        total_dataset = pickle.load(f)
    test_data = total_dataset["test"]
    test_size = len(test_data[args.text_key])
    if args.max_test_samples:
        test_size = min(test_size, args.max_test_samples)
    if args.num_candidates > test_size:
        args.num_candidates = max(1, test_size - 1)
        print(f"Reduced num_candidates to {args.num_candidates}")

    print(f"Reconstructing audio paths from {args.train_json}...")
    audio_paths = _load_audio_paths(args.train_json, args.wav_dir, test_size)
    if not audio_paths:
        raise SystemExit("No audio paths found — check --train-json and --wav-dir")
    print(f"Found {len(audio_paths)} audio files")

    # --- Load ambient noise files ---
    ambient_files: list[Path] | None = None
    if args.ambient_dir:
        ambient_files = _load_ambient_files(Path(args.ambient_dir), args.ambient_source)
        print(f"Loaded {len(ambient_files)} ambient clips from {args.ambient_dir}")

    # --- Load models ---
    print("Loading CLASP model...")
    clasp_model = load_model(args.model_path, device)

    print("Loading HuBERT...")
    hubert_processor = AutoProcessor.from_pretrained(args.hubert_model)
    hubert_model = HubertModel.from_pretrained(args.hubert_model).to(device)
    hubert_model.eval()

    print("Loading EfficientNet-B7...")
    vision_model, vision_preprocess = load_efficientnet_b7(device)

    # --- Clean baseline (from stored pkl embeddings) ---
    print("\nComputing clean baseline...")
    test_metadata = build_test_metadata(test_size, args.num_candidates)
    clean_loader = DataLoader(
        TestDataset(test_data, test_metadata, args.audio_key, args.text_key),
        batch_size=1, shuffle=False,
    )
    results_clean = evaluate_model_on_candidates(clasp_model, clean_loader, device, threshold=0.5)
    print(f"  Clean  Hits@1: {results_clean['Hits@1']:.4f}  MRR: {results_clean['MRR']:.4f}")

    if use_wandb:
        wandb.log({"clean/" + k: v for k, v in results_clean.items()})

    # --- Setup results tracking ---
    snr_levels = [float(x) for x in args.snr_levels.split(",")]
    drr_levels = [float(x) for x in args.drr_levels.split(",")]

    all_results: dict[str, dict] = {"clean": results_clean}
    metric_keys: list[str] = list(results_clean.keys())

    results_table: wandb.Table | None = None
    if use_wandb:
        results_table = wandb.Table(
            columns=["noise_config", "noise_type", "level_axis", "level"] + metric_keys
        )
        results_table.add_data("clean", "clean", "—", None,
                               *[results_clean[k] for k in metric_keys])

    # Shared kwargs for _run_condition
    shared = dict(
        audio_paths=audio_paths,
        hubert_processor=hubert_processor,
        hubert_model=hubert_model,
        vision_model=vision_model,
        vision_preprocess=vision_preprocess,
        device=device,
        vision_batch_size=args.vision_batch_size,
        test_data=test_data,
        test_metadata=test_metadata,
        audio_key=args.audio_key,
        text_key=args.text_key,
        clasp_model=clasp_model,
        all_results=all_results,
        results_table=results_table,
        metric_keys=metric_keys,
        use_wandb=use_wandb,
    )

    # --- White noise (SNR sweep) ---
    print("\n=== White Noise ===")
    for snr in snr_levels:
        _run_condition(
            label=f"white_snr{snr:+.0f}",
            noise_type="white",
            level=snr,
            level_axis="snr_db",
            ambient_files=None,
            **shared,
        )

    # --- Ambient noise (SNR sweep) ---
    if ambient_files is not None:
        print("\n=== Ambient Noise ===")
        for snr in snr_levels:
            _run_condition(
                label=f"ambient_snr{snr:+.0f}",
                noise_type="ambient",
                level=snr,
                level_axis="snr_db",
                ambient_files=ambient_files,
                **shared,
            )
    else:
        print("\n[Skipping ambient noise — provide --ambient-dir to enable]")

    # --- Reverberation (DRR sweep) ---
    print("\n=== Reverberation ===")
    for drr in drr_levels:
        _run_condition(
            label=f"reverb_drr{drr:+.0f}",
            noise_type="reverb",
            level=drr,
            level_axis="drr_db",
            ambient_files=None,
            **shared,
        )

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("NOISE ROBUSTNESS EVALUATION RESULTS")
    print("=" * 80)
    for config, m in all_results.items():
        print(f"  {config:<30}  Hits@1={m['Hits@1']:.4f}  MRR={m['MRR']:.4f}")

    # --- W&B finish ---
    if use_wandb:
        noisy_hits = [m["Hits@1"] for k, m in all_results.items() if k != "clean"]
        wandb.summary["clean_Hits@1"] = results_clean["Hits@1"]
        wandb.summary["clean_MRR"] = results_clean["MRR"]
        wandb.summary["min_noisy_Hits@1"] = min(noisy_hits)
        wandb.summary["max_noisy_Hits@1"] = max(noisy_hits)
        wandb.summary["avg_relative_degradation"] = (
            1 - (sum(noisy_hits) / len(noisy_hits)) / results_clean["Hits@1"]
        )
        wandb.log({"results_table": results_table})
        wandb.finish()

    # --- CSV output ---
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["noise_config"] + metric_keys)
            writer.writeheader()
            for config, m in all_results.items():
                writer.writerow({"noise_config": config, **m})
        print(f"\nSaved results to {args.output_csv}")


if __name__ == "__main__":
    main()
