#!/usr/bin/env python3
"""Build total_dataset .pkl from SpeechBrown-style global_metadata.json + local audio files.

Each metadata row needs `text` and a relative audio path in `audio_file_path` or `file_path`
(e.g. SpeechBrown `global_metadata.json` uses `file_path`).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoProcessor, HubertModel
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.config.settings import get_default_device
from clasp.data.speechbrown_paths import (
    audio_relpath as _audio_relpath,
    load_metadata_entries as _load_metadata_entries,
    resolve_existing_audio_file as _resolve_existing_audio_file,
)
from clasp.inference.audio_preprocess import audio_duration_seconds
from clasp.inference.embed_audio import hubert_audio_files
from clasp.inference.spectrogram_image import (
    efficientnet_embeddings_from_audio_paths,
    load_efficientnet_b7,
)


def _squeeze_hubert_list(embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
    out = []
    for e in embeddings:
        if e.dim() > 1:
            e = e.squeeze(0)
        out.append(e.contiguous().float().cpu())
    return out


def _split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-3:
        raise ValueError(f"train + val + test must sum to 1.0, got {s}")
    indices = list(range(n))
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )
    rel_val = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=rel_val,
        random_state=seed,
        shuffle=True,
    )
    return train_idx, val_idx, test_idx


def _build_split_dict(
    rows: list[dict],
    indices: list[int],
    hubert_processor,
    hubert_model,
    sentence_model: SentenceTransformer,
    vision_model,
    vision_preprocess,
    device: torch.device,
    vision_batch_size: int,
    text_batch_size: int,
) -> dict:
    paths = [str(rows[i]["_abs_audio"]) for i in indices]
    texts = [rows[i]["text"] for i in indices]
    sources = None
    if all("category" in rows[i] for i in indices):
        sources = [str(rows[i].get("category", "")) for i in indices]

    hubert_raw = hubert_audio_files(paths, hubert_processor, hubert_model, device)
    hubert_list = _squeeze_hubert_list(hubert_raw)

    text_emb = sentence_model.encode(
        texts,
        batch_size=text_batch_size,
        convert_to_tensor=True,
        show_progress_bar=len(texts) > 32,
    )
    text_list = []
    for j in range(text_emb.size(0)):
        t = text_emb[j].detach().cpu().float()
        if t.dim() > 1:
            t = t.squeeze(0)
        text_list.append(t)

    image_list = efficientnet_embeddings_from_audio_paths(
        paths,
        vision_model,
        vision_preprocess,
        device,
        batch_size=vision_batch_size,
    )
    image_list = [t.squeeze(0) if t.dim() > 1 else t for t in image_list]

    split = {
        "hubert-emb": hubert_list,
        "text": text_list,
        "image": image_list,
    }
    if sources is not None:
        split["source"] = sources
    return split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata-json", type=Path, required=True, help="Path to global_metadata.json")
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root used to resolve relative audio paths (audio_file_path or file_path in JSON).",
    )
    p.add_argument("--output", type=Path, default=Path("data/datasets/total_dataset_real.pkl"))
    p.add_argument("--max-samples", type=int, default=None, help="Cap number of usable rows after filtering.")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda, cuda:0, or cpu (default: auto).")
    p.add_argument("--hubert-model", default="facebook/hubert-large-ls960-ft")
    p.add_argument("--sentence-transformer", default="sentence-transformers/LaBSE")
    p.add_argument("--vision-batch-size", type=int, default=4)
    p.add_argument("--text-batch-size", type=int, default=32)
    p.add_argument(
        "--min-audio-seconds",
        type=float,
        default=0.0,
        help="Skip rows whose audio file is shorter than this (0 = no minimum).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()

    entries = _load_metadata_entries(args.metadata_json)
    rows: list[dict] = []
    for e in tqdm(entries, desc="filter metadata"):
        rel_audio = _audio_relpath(e)
        if rel_audio is None or "text" not in e:
            continue
        abs_audio = _resolve_existing_audio_file(args.dataset_root, rel_audio)
        if abs_audio is None:
            continue
        if args.min_audio_seconds > 0:
            try:
                if audio_duration_seconds(abs_audio) < args.min_audio_seconds:
                    continue
            except OSError:
                continue
        row = dict(e)
        row["_abs_audio"] = str(abs_audio)
        rows.append(row)
        if args.max_samples is not None and len(rows) >= args.max_samples:
            break

    if not rows:
        raise SystemExit(
            "No usable rows: unzip dataset_part*.zip under --dataset-root, ensure WAVs exist, "
            "and that paths in JSON (file_path) match on disk (dataset/part1 vs dataset_part1 is tried automatically)."
        )

    n = len(rows)
    train_idx, val_idx, test_idx = _split_indices(
        n, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    test_n = len(test_idx)
    if test_n < 100:
        warnings.warn(
            f"Test split has only {test_n} samples. "
            "run_retrieval_eval.py defaults to --num-candidates 100; use --num-candidates <= "
            f"{max(1, test_n - 1)} or increase data.",
            stacklevel=1,
        )

    print(f"Using {n} samples: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} on {device}")

    hubert_processor = AutoProcessor.from_pretrained(args.hubert_model)
    hubert_model = HubertModel.from_pretrained(args.hubert_model).to(device)
    hubert_model.eval()

    sentence_model = SentenceTransformer(args.sentence_transformer, device=str(device))

    vision_model, vision_preprocess = load_efficientnet_b7(device)

    def build(indices: list[int]) -> dict:
        return _build_split_dict(
            rows,
            indices,
            hubert_processor,
            hubert_model,
            sentence_model,
            vision_model,
            vision_preprocess,
            device,
            args.vision_batch_size,
            args.text_batch_size,
        )

    total_dataset = {
        "train": build(train_idx),
        "validation": build(val_idx),
        "test": build(test_idx),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(total_dataset, f)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
