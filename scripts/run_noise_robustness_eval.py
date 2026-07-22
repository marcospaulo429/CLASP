#!/usr/bin/env python3
"""Noise robustness evaluation paragraph-aware (Spoken-SQuAD).

Lê um PKL produzido por ``scripts/build_spoken_squad_pkl.py`` (com chave
``audio_paths`` e opcional ``paragraph_id`` + ``_meta.pooling_mode``), aplica
ruído (white / ambient / reverb) ao waveform e reextrai embeddings HuBERT +
EfficientNet, depois roda retrieval e compara com o baseline limpo.

* PKL ``mean``    → ``audio_paths[i]`` é a lista de WAVs do parágrafo i, que
                    são concatenados antes do noise/embedding. Retrieval usa
                    ``evaluate_model_on_candidates`` no split inteiro.
* PKL ``chunked`` → ``audio_paths[i]`` tem 1 WAV (o chunk i). Retrieval usa
                    ``evaluate_model_on_paragraph_groups`` (max-sim por
                    paragraph_id).
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, HubertModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.audio.noise_augmentation import (
    add_ambient_noise,
    add_reverberation,
    add_white_noise,
    scan_esc50_files,
)
from clasp.config.settings import get_default_device
from clasp.data.datasets import TestDataset, build_test_metadata
from clasp.evaluation.metrics import (
    evaluate_model_on_candidates,
    evaluate_model_on_paragraph_groups,
)
from clasp.inference.audio_preprocess import load_mono_16k_padded
from clasp.inference.embed_audio import hubert_numpy_waveform
from clasp.inference.pipeline import load_model
from clasp.inference.spectrogram_image import (
    efficientnet_embedding_from_waveform,
    load_efficientnet_b7,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-path", type=Path, required=True,
                   help="Pickle Spoken-SQuAD (com audio_paths)")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--audio-key", default="hubert-emb")
    p.add_argument("--text-key", default="text")
    p.add_argument("--snr-levels", default="20,15,10,5",
                   help="dB; comma-separated")
    p.add_argument(
        "--noise-types",
        default="white,reverb",
        help="Subset of {white, ambient, reverb}; ambient requires --wham-dir or --esc50-dir.",
    )
    p.add_argument("--wham-dir", type=Path, default=None,
                   help="WHAM dir with noise/*.wav (used for ambient noise; alt: --esc50-dir).")
    p.add_argument("--esc50-dir", type=Path, default=None,
                   help="ESC-50 dir with audio/*.wav (used for ambient noise; alt: --wham-dir).")
    p.add_argument("--ambient-num-samples", type=int, default=64,
                   help="How many ambient WAVs to pre-load and sample from (per-row).")
    p.add_argument("--ambient-seed", type=int, default=42,
                   help="Seed for per-row ambient noise selection (reproducibility).")
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--hubert-model", default="facebook/hubert-large-ls960-ft")
    p.add_argument("--chunk-samples", type=int, default=320_000)
    p.add_argument("--chunk-batch-size", type=int, default=4)
    p.add_argument("--retrieval-batch-size", type=int, default=64,
                   help="Batch da fusão CLASP no eval paragraph_grouped")
    p.add_argument("--hits-k", default="1,5,10,50")
    return p.parse_args()


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_str(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_concat(paths: list[str]) -> np.ndarray:
    pieces = [load_mono_16k_padded(p) for p in paths]
    if not pieces:
        return np.zeros(16_000, dtype=np.float32)
    return np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in pieces])


def _load_ambient_pool(
    wham_dir: Path | None,
    esc50_dir: Path | None,
    n_samples: int,
    seed: int,
) -> list[np.ndarray]:
    """Pre-load a pool of ambient noise WAVs from WHAM and/or ESC-50."""
    candidates: list[Path] = []
    if wham_dir is not None:
        candidates.extend(sorted((wham_dir / "noise").glob("*.wav")))
    if esc50_dir is not None:
        candidates.extend(scan_esc50_files(esc50_dir))
    if not candidates:
        raise FileNotFoundError(
            f"No ambient noise WAVs found (wham_dir={wham_dir}, esc50_dir={esc50_dir})"
        )
    rng = np.random.default_rng(seed)
    if len(candidates) > n_samples:
        idx = rng.choice(len(candidates), n_samples, replace=False)
        candidates = [candidates[i] for i in idx]
    return [load_mono_16k_padded(str(p)) for p in candidates]


def _apply_noise(
    y: np.ndarray,
    noise_type: str,
    snr: float,
    ambient_sample: np.ndarray | None,
) -> np.ndarray:
    if noise_type == "white":
        return add_white_noise(y, snr)
    if noise_type == "ambient":
        if ambient_sample is None:
            raise ValueError("ambient noise requires --wham-dir or --esc50-dir")
        return add_ambient_noise(y, ambient_sample, snr)
    if noise_type == "reverb":
        # use SNR as a *strength* knob: tempo de decaimento em ms = snr*10
        return add_reverberation(y, decay_time_ms=snr * 10)
    raise ValueError(f"unknown noise_type {noise_type!r}")


def _reembed_all_conditions(
    audio_paths_per_row: list[list[str]],
    cond_keys: list[tuple[str, float]],
    *,
    ambient_pool: list[np.ndarray] | None,
    ambient_idx_per_row: np.ndarray | None,
    hubert_processor,
    hubert_model,
    vision_model,
    vision_preprocess,
    device: torch.device,
    chunk_samples: int,
    chunk_batch_size: int,
) -> dict[tuple[str, float], tuple[list[torch.Tensor], list[torch.Tensor]]]:
    """Single pass over rows; loads each wav once, applies every (noise_type, snr).

    Returns a dict mapping (noise_type, snr) -> (hubert_list, image_list).
    """
    from tqdm import tqdm

    out: dict[tuple[str, float], tuple[list[torch.Tensor], list[torch.Tensor]]] = {
        k: ([], []) for k in cond_keys
    }

    for row_idx, paths in enumerate(tqdm(audio_paths_per_row, desc="re-embed (all conds)")):
        y_clean = _load_concat(paths)
        amb = (
            ambient_pool[int(ambient_idx_per_row[row_idx])]
            if ambient_pool is not None and ambient_idx_per_row is not None
            else None
        )
        for nt, snr in cond_keys:
            y_noisy = _apply_noise(y_clean, nt, snr, amb).astype(np.float32)
            h = hubert_numpy_waveform(
                y_noisy, hubert_processor, hubert_model, device,
                chunk_samples=chunk_samples, chunk_batch_size=chunk_batch_size,
                pooling="mean",
            )
            s = efficientnet_embedding_from_waveform(
                y_noisy, vision_model, vision_preprocess, device,
                chunk_samples=chunk_samples, chunk_batch_size=chunk_batch_size,
                pooling="mean",
            )
            out[(nt, snr)][0].append(h.detach().cpu().float().contiguous())
            out[(nt, snr)][1].append(s.detach().cpu().float().contiguous())

    return out


def _retrieve(
    model,
    test_data: dict,
    *,
    is_chunked: bool,
    audio_key: str,
    text_key: str,
    device: torch.device,
    ks: list[int],
    batch_size: int,
) -> dict:
    if is_chunked:
        return evaluate_model_on_paragraph_groups(
            model, test_data, device,
            audio_key=audio_key, text_key=text_key,
            ks=ks, batch_size=batch_size,
        )
    n = len(test_data[text_key])
    metadata = build_test_metadata(n, n)  # split inteiro como pool
    ds = TestDataset(test_data, metadata, audio_key, text_key)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    return evaluate_model_on_candidates(model, loader, device, threshold=0.5)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()

    with open(args.dataset_path, "rb") as f:
        total = pickle.load(f)

    test_data = total.get("test") or total.get("validation")
    if test_data is None:
        raise SystemExit("PKL has neither 'test' nor 'validation' split")

    # Spoken-SQuAD format: audio_paths é lista de listas
    # if "audio_paths" not in test_data:
    #     raise SystemExit(
    #         "PKL test split is missing 'audio_paths'. Re-build with the new "
    #         "scripts/build_spoken_squad_pkl.py."
    #     )
    # audio_paths_per_row: list[list[str]] = list(test_data["audio_paths"])

    # VoxPopuli format: audio_path é lista simples — wrapeamos cada item numa lista
    if "audio_paths" in test_data:
        audio_paths_per_row: list[list[str]] = list(test_data["audio_paths"])
    elif "audio_path" in test_data:
        audio_paths_per_row = [[p] for p in test_data["audio_path"]]
    else:
        raise SystemExit(
            "PKL test split is missing 'audio_paths' (Spoken-SQuAD) ou 'audio_path' (VoxPopuli)."
        )

    pooling_mode = (total.get("_meta") or {}).get("pooling_mode")
    is_chunked = pooling_mode == "chunked" or "paragraph_id" in test_data and all(
        len(p) == 1 for p in audio_paths_per_row
    )
    print(f"PKL pooling_mode={pooling_mode!r}  -> retrieval mode={'paragraph_grouped' if is_chunked else 'candidate'}")

    snr_levels = _parse_csv_floats(args.snr_levels)
    noise_types = _parse_csv_str(args.noise_types)
    ks = _parse_csv_ints(args.hits_k)

    ambient_pool: list[np.ndarray] | None = None
    ambient_idx_per_row: np.ndarray | None = None
    if "ambient" in noise_types:
        if args.wham_dir is None and args.esc50_dir is None:
            print("Skipping ambient noise (no --wham-dir / --esc50-dir).")
            noise_types = [n for n in noise_types if n != "ambient"]
        else:
            try:
                ambient_pool = _load_ambient_pool(
                    args.wham_dir, args.esc50_dir,
                    n_samples=args.ambient_num_samples, seed=args.ambient_seed,
                )
                rng = np.random.default_rng(args.ambient_seed)
                ambient_idx_per_row = rng.integers(
                    0, len(ambient_pool), size=len(audio_paths_per_row)
                )
                print(f"Ambient pool: {len(ambient_pool)} WAVs pre-loaded.")
            except FileNotFoundError as e:
                print(f"Ambient noise unavailable ({e}); skipping ambient.")
                noise_types = [n for n in noise_types if n != "ambient"]

    # Models
    print("Loading CLASP model …")
    model = load_model(args.model_path, device)
    print("Loading HuBERT + EfficientNet …")
    hubert_processor = AutoProcessor.from_pretrained(args.hubert_model)
    hubert_model = HubertModel.from_pretrained(args.hubert_model).to(device).eval()
    vision_model, vision_preprocess = load_efficientnet_b7(device)

    # Clean baseline using existing pickle embeddings
    print("Computing CLEAN baseline …")
    results: dict[str, dict] = {
        "clean": _retrieve(
            model, test_data,
            is_chunked=is_chunked,
            audio_key=args.audio_key, text_key=args.text_key,
            device=device, ks=ks, batch_size=args.retrieval_batch_size,
        )
    }
    print(f"  clean: {results['clean']}")

    cond_keys: list[tuple[str, float]] = [(nt, snr) for nt in noise_types for snr in snr_levels]
    if cond_keys:
        print(f"\nRe-embedding {len(cond_keys)} noise conditions in a single row-major pass …")
        reembedded = _reembed_all_conditions(
            audio_paths_per_row, cond_keys,
            ambient_pool=ambient_pool, ambient_idx_per_row=ambient_idx_per_row,
            hubert_processor=hubert_processor, hubert_model=hubert_model,
            vision_model=vision_model, vision_preprocess=vision_preprocess,
            device=device,
            chunk_samples=args.chunk_samples, chunk_batch_size=args.chunk_batch_size,
        )

        for nt, snr in cond_keys:
            tag = f"{nt}_{snr}"
            new_audio, new_image = reembedded[(nt, snr)]
            new_test = dict(test_data)
            new_test[args.audio_key] = new_audio
            new_test["image"] = new_image
            r = _retrieve(
                model, new_test,
                is_chunked=is_chunked,
                audio_key=args.audio_key, text_key=args.text_key,
                device=device, ks=ks, batch_size=args.retrieval_batch_size,
            )
            results[tag] = r
            print(f"  {tag}: Hits@1={r.get('Hits@1', 'NA')}  MRR={r.get('MRR', 'NA')}")

    print("\n" + "=" * 60)
    print("NOISE ROBUSTNESS RESULTS")
    print("=" * 60)
    for tag, r in results.items():
        print(f"{tag}: {r}")

    if args.output_csv:
        all_keys: list[str] = []
        for r in results.values():
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["noise_config", *all_keys])
            writer.writeheader()
            for tag, r in results.items():
                writer.writerow({"noise_config": tag, **{k: r.get(k) for k in all_keys}})
        print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
