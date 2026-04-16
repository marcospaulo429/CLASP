#!/usr/bin/env python3
"""Noise robustness evaluation: measure retrieval performance degradation under noise.

Adds white noise, ambient noise, and reverberation to test audio samples,
then compares retrieval metrics (Hits@1, MRR, etc.) against clean baseline.
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, HubertModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.audio.noise_augmentation import add_ambient_noise, add_reverberation, add_white_noise
from clasp.config.settings import get_default_device
from clasp.data.datasets import TestDataset, build_test_metadata
from clasp.evaluation.metrics import evaluate_model_on_candidates
from clasp.inference.audio_preprocess import load_mono_16k_padded, TARGET_SR
from clasp.inference.embed_audio import hubert_audio_files
from clasp.inference.pipeline import load_model
from clasp.inference.spectrogram_image import (
    efficientnet_embeddings_from_audio_paths,
    load_efficientnet_b7,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, help="Path to total_dataset_spoken_squad.pkl")
    parser.add_argument("--model-path", required=True, help="Path to CLASP checkpoint")
    parser.add_argument(
        "--train-json",
        default=Path("data/datasets/spoken_squad/spoken_train-v1.1.json"),
        help="Spoken SQuAD JSON to reconstruct audio paths",
    )
    parser.add_argument(
        "--wav-dir",
        default=Path("data/datasets/spoken_squad/train_wav"),
        help="Directory with WAV files",
    )
    parser.add_argument("--audio-key", default="hubert-emb")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--num-candidates", type=int, default=10, help="Number of candidates for eval")
    parser.add_argument(
        "--snr-levels",
        default="20,15,10,5",
        help="SNR levels in dB (comma-separated, e.g. '20,15,10,5')",
    )
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu (default: auto)")
    parser.add_argument(
        "--wham-dir",
        default=None,
        help="Directory with WHAM ambient noise files. If None, skip ambient noise eval.",
    )
    parser.add_argument("--output-csv", default=None, help="Path to save results as CSV")
    parser.add_argument("--hubert-model", default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--vision-batch-size", type=int, default=4)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--max-test-samples", type=int, default=None, help="Limit test set size")
    return parser.parse_args()


def load_spoken_squad_audio_paths(json_path: Path, wav_dir: Path, num_samples: int | None = None):
    """Reconstruct audio file paths from Spoken SQuAD JSON.
    
    Returns:
        List of absolute audio paths in the same order as parsed from JSON.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    articles = data["data"]
    paths = []

    for a_idx, article in enumerate(articles):
        for p_idx, para in enumerate(article["paragraphs"]):
            for q_idx, qa in enumerate(para["qas"]):
                wav_path = wav_dir / f"{a_idx}_{p_idx}_{q_idx}.wav"
                if wav_path.is_file():
                    paths.append(str(wav_path))
                    if num_samples is not None and len(paths) >= num_samples:
                        return paths
    
    return paths


def load_wham_sample(wham_dir: Path, index: int = 0) -> np.ndarray:
    """Load a WHAM ambient noise sample.
    
    Returns:
        Audio as float32 array at 16 kHz.
    """
    wham_files = sorted((wham_dir / "noise").glob("*.wav"))[:5]  # Use first 5 files
    if not wham_files:
        raise FileNotFoundError(f"No WHAM noise files found in {wham_dir}/noise")
    
    noise_file = wham_files[index % len(wham_files)]
    return load_mono_16k_padded(noise_file)


def build_noisy_embeddings(
    audio_paths: list[str],
    noise_type: str,
    snr_db: float,
    hubert_processor,
    hubert_model,
    vision_model,
    vision_preprocess,
    device: torch.device,
    wham_audio: np.ndarray | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Load audio, add noise, and re-embed using HuBERT and vision model.
    
    Returns:
        Tuple of (hubert_embeddings, vision_embeddings)
    """
    hubert_embeddings = []
    vision_audios = []  # Store noisy audio for vision model
    
    for file_path in audio_paths:
        # Load clean audio
        audio = load_mono_16k_padded(file_path)
        
        # Add noise
        if noise_type == "white":
            noisy_audio = add_white_noise(audio, snr_db)
        elif noise_type == "ambient":
            if wham_audio is None:
                raise ValueError("wham_audio required for ambient noise")
            noisy_audio = add_ambient_noise(audio, wham_audio, snr_db)
        elif noise_type == "reverb":
            noisy_audio = add_reverberation(audio, decay_time_ms=snr_db * 10)  # Use SNR param for decay time
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # HuBERT embedding
        audio_tensor = torch.from_numpy(noisy_audio.astype(np.float32))
        inputs = hubert_processor(audio_tensor, sampling_rate=TARGET_SR, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden_states = hubert_model(**inputs).last_hidden_state
            avg_embedding = torch.mean(hidden_states, dim=1)
            hubert_embeddings.append(avg_embedding.squeeze(0).cpu())
        
        vision_audios.append(file_path)  # Temporary: use original path, but we'd need to save noisy audio
    
    # Vision embeddings (use efficientnet_embeddings_from_audio_paths)
    # Note: This loads from disk, so for now we use clean audio for vision
    # In a full impl, we'd save noisy audio and load from there
    vision_embeddings = efficientnet_embeddings_from_audio_paths(
        vision_audios, vision_model, vision_preprocess, device, batch_size=4
    )
    
    return hubert_embeddings, vision_embeddings


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()
    
    # Load pickle
    with open(args.dataset_path, "rb") as f:
        total_dataset = pickle.load(f)
    
    test_data = total_dataset["test"]
    test_size = len(test_data[args.text_key])
    
    if args.max_test_samples:
        test_size = min(test_size, args.max_test_samples)
    
    if args.num_candidates > test_size:
        print(f"Warning: num_candidates ({args.num_candidates}) > test_size ({test_size})")
        print(f"Reducing num_candidates to {max(1, test_size - 1)}")
        args.num_candidates = max(1, test_size - 1)
    
    # Reconstruct audio paths from JSON
    print(f"Reconstructing audio paths from {args.train_json}...")
    audio_paths = load_spoken_squad_audio_paths(args.train_json, args.wav_dir, test_size)
    
    if not audio_paths:
        raise SystemExit("No audio paths found")
    
    print(f"Found {len(audio_paths)} audio files for test set")
    
    # Load models
    print("Loading CLASP model...")
    model = load_model(args.model_path, device)
    
    print("Loading HuBERT and vision models...")
    hubert_processor = AutoProcessor.from_pretrained(args.hubert_model)
    hubert_model = HubertModel.from_pretrained(args.hubert_model).to(device)
    hubert_model.eval()
    
    vision_model, vision_preprocess = load_efficientnet_b7(device)
    
    # Load WHAM if needed
    wham_audio = None
    if args.wham_dir:
        print(f"Loading WHAM ambient noise from {args.wham_dir}...")
        try:
            wham_audio = load_wham_sample(Path(args.wham_dir))
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            args.wham_dir = None
    
    # Clean baseline (use embeddings from pickle)
    print("Computing clean baseline metrics...")
    test_metadata = build_test_metadata(test_size, args.num_candidates)
    test_dataset_clean = TestDataset(test_data, test_metadata, args.audio_key, args.text_key)
    from torch.utils.data import DataLoader
    test_loader_clean = DataLoader(test_dataset_clean, batch_size=1, shuffle=False)
    results_clean = evaluate_model_on_candidates(model, test_loader_clean, device, threshold=0.5)
    
    # Parse SNR levels
    snr_levels = [float(x.strip()) for x in args.snr_levels.split(",")]
    
    # Dictionary to store all results
    all_results = {"clean": results_clean}
    
    # Evaluate under noise
    noise_types_to_test = ["white"]
    if args.wham_dir:
        noise_types_to_test.append("ambient")
    noise_types_to_test.append("reverb")
    
    for noise_type in noise_types_to_test:
        for snr in snr_levels:
            noise_name = f"{noise_type}_{snr}"
            print(f"\nEvaluating: {noise_name}")
            
            # Build noisy embeddings
            hubert_embs, vision_embs = build_noisy_embeddings(
                audio_paths,
                noise_type,
                snr,
                hubert_processor,
                hubert_model,
                vision_model,
                vision_preprocess,
                device,
                wham_audio=wham_audio,
            )
            
            # Create test dataset with noisy embeddings
            test_data_noisy = {
                args.audio_key: hubert_embs,
                "image": vision_embs,
                args.text_key: test_data[args.text_key],
            }
            
            test_dataset_noisy = TestDataset(test_data_noisy, test_metadata, args.audio_key, args.text_key)
            test_loader_noisy = DataLoader(test_dataset_noisy, batch_size=1, shuffle=False)
            results_noisy = evaluate_model_on_candidates(model, test_loader_noisy, device, threshold=0.5)
            
            all_results[noise_name] = results_noisy
            print(f"  Hits@1: {results_noisy['Hits@1']:.4f}, MRR: {results_noisy['MRR']:.4f}")
    
    # Print results table
    print("\n" + "=" * 80)
    print("NOISE ROBUSTNESS EVALUATION RESULTS")
    print("=" * 80)
    
    for noise_config, metrics in all_results.items():
        print(f"\n{noise_config}:")
        print(f"  Hits@1:  {metrics['Hits@1']:.4f}")
        print(f"  MRR:     {metrics['MRR']:.4f}")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    
    # Save to CSV if requested
    if args.output_csv:
        print(f"\nSaving results to {args.output_csv}...")
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["noise_config"] + list(results_clean.keys()))
            writer.writeheader()
            for noise_config, metrics in all_results.items():
                row = {"noise_config": noise_config}
                row.update(metrics)
                writer.writerow(row)


if __name__ == "__main__":
    main()
