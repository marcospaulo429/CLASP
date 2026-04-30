#!/usr/bin/env python3
"""Build total_dataset .pkl from Spoken SQuAD JSON + local WAV files.

Spoken SQuAD JSON structure:
  {"data": [{"title": "...", "paragraphs": [{"context": "...", "qas": [{"id": "...", "question": "..."}]}]}]}

WAV files are named by position: {article_idx}_{paragraph_idx}_{qa_idx}.wav
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
import numpy as np

import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoProcessor, HubertModel
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.config.settings import get_default_device
from clasp.inference.audio_preprocess import load_mono_16k_padded
from clasp.inference.embed_audio import hubert_audio_files
from clasp.inference.spectrogram_image import (
    efficientnet_embeddings_from_audio_paths,
    load_efficientnet_b7,
)
from clasp.audio.noise_augmentation import (
    add_ambient_noise,
    add_reverberation,
    add_white_noise,
    load_esc50_clip,
    scan_esc50_files,
)


def _squeeze_hubert_list(embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
    out = []
    for e in embeddings:
        if e.dim() > 1:
            e = e.squeeze(0)
        out.append(e.contiguous().float().cpu())
    return out


def hubert_audio_files_with_noise(
    paths: list[str],
    processor,
    model,
    device: torch.device,
    noise_prob: float,
    noise_snr: float,
    noise_types: list[str],
    esc50_files: list | None = None,
) -> list[torch.Tensor]:
    embeddings = []
    for path in tqdm(paths, desc="hubert+noise"):
        audio = load_mono_16k_padded(path)
        if noise_prob > 0.0 and np.random.random() < noise_prob:
            noise_type = np.random.choice(noise_types)
            if noise_type == "white":
                audio = add_white_noise(audio, snr_db=noise_snr)
            elif noise_type == "reverb":
                audio = add_reverberation(audio, sr=16000)
            elif noise_type == "ambient" and esc50_files:
                noise_clip = load_esc50_clip(esc50_files, target_sr=16000)
                audio = add_ambient_noise(audio, noise_clip, snr_db=noise_snr)
        t = torch.from_numpy(audio.astype(np.float32))
        inputs = processor(t, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
            embeddings.append(torch.mean(hidden, dim=1))
    return embeddings


def _split_indices(
    n: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
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


def load_spoken_squad_rows(json_path: Path, wav_dir: Path, max_samples: int | None) -> list[dict]:
    """Parse Spoken SQuAD JSON and pair each QA entry with its positional WAV file."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    articles = data["data"]
    rows: list[dict] = []

    for a_idx, article in enumerate(tqdm(articles, desc="parse JSON")):
        for p_idx, para in enumerate(article["paragraphs"]):
            for q_idx, qa in enumerate(para["qas"]):
                wav_path = wav_dir / f"{a_idx}_{p_idx}_{q_idx}.wav"
                if not wav_path.is_file():
                    continue
                rows.append({
                    "text": qa["question"],
                    "_abs_audio": str(wav_path),
                })
                if max_samples is not None and len(rows) >= max_samples:
                    return rows

    return rows


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
    noise_prob: float = 0.0,
    noise_snr: float = 20.0,
    noise_types: list[str] | None = None,
    esc50_files: list | None = None,
) -> dict:
    paths = [rows[i]["_abs_audio"] for i in indices]
    texts = [rows[i]["text"] for i in indices]

    if noise_prob > 0.0 and noise_types:
        hubert_raw = hubert_audio_files_with_noise(
            paths, hubert_processor, hubert_model, device,
            noise_prob, noise_snr, noise_types, esc50_files,
        )
    else:
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

    return {
        "hubert-emb": hubert_list,
        "text": text_list,
        "image": image_list,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-json",
        type=Path,
        default=Path("data/datasets/spoken_squad/spoken_train-v1.1.json"),
        help="Path to spoken_train-v1.1.json",
    )
    p.add_argument(
        "--wav-dir",
        type=Path,
        default=Path("data/datasets/spoken_squad/train_wav"),
        help="Directory containing WAV files named {a}_{p}_{q}.wav",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/total_dataset_spoken_squad.pkl"),
    )
    p.add_argument("--max-samples", type=int, default=None, help="Cap rows (useful for quick smoke tests).")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda, cuda:0, or cpu (default: auto).")
    p.add_argument("--hubert-model", default="facebook/hubert-large-ls960-ft")
    p.add_argument("--sentence-transformer", default="sentence-transformers/LaBSE")
    p.add_argument("--vision-batch-size", type=int, default=4)
    p.add_argument("--text-batch-size", type=int, default=32)
    # Noise augmentation (applied only to training split)
    p.add_argument("--noise-prob", type=float, default=0.0,
                   help="Probability [0-1] of applying noise to each training sample.")
    p.add_argument("--noise-snr", type=float, default=20.0,
                   help="SNR in dB for noise augmentation (lower = more noise).")
    p.add_argument("--noise-types", nargs="+", default=["white", "reverb"],
                   choices=["white", "reverb", "ambient"],
                   help="Noise types to randomly sample from.")
    p.add_argument("--esc50-dir", type=Path, default=None,
                   help="Path to ESC-50 dataset root (required when 'ambient' is in --noise-types).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()

    rows = load_spoken_squad_rows(args.train_json, args.wav_dir, args.max_samples)

    if not rows:
        raise SystemExit(
            "No usable rows found. Check that WAV files exist under --wav-dir "
            "and that --train-json points to a valid spoken_train-v1.1.json."
        )

    n = len(rows)
    train_idx, val_idx, test_idx = _split_indices(
        n, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    test_n = len(test_idx)
    if test_n < 100:
        warnings.warn(
            f"Test split has only {test_n} samples. "
            "run_retrieval_eval.py defaults to --num-candidates 100; pass "
            f"--num-candidates <= {max(1, test_n - 1)} or increase data.",
            stacklevel=1,
        )

    print(f"Using {n} rows → train={len(train_idx)}, val={len(val_idx)}, test={test_n} on {device}")

    hubert_processor = AutoProcessor.from_pretrained(args.hubert_model)
    hubert_model_obj = HubertModel.from_pretrained(args.hubert_model).to(device)
    hubert_model_obj.eval()

    sentence_model = SentenceTransformer(args.sentence_transformer, device=str(device))

    vision_model, vision_preprocess = load_efficientnet_b7(device)

    esc50_files = None
    if "ambient" in args.noise_types:
        if args.esc50_dir is None:
            raise SystemExit("--esc50-dir is required when 'ambient' is in --noise-types.")
        esc50_files = scan_esc50_files(args.esc50_dir)
        print(f"Loaded {len(esc50_files)} ESC-50 clips from {args.esc50_dir}")

    def build(indices: list[int], augment: bool = False) -> dict:
        return _build_split_dict(
            rows,
            indices,
            hubert_processor,
            hubert_model_obj,
            sentence_model,
            vision_model,
            vision_preprocess,
            device,
            args.vision_batch_size,
            args.text_batch_size,
            noise_prob=args.noise_prob if augment else 0.0,
            noise_snr=args.noise_snr,
            noise_types=args.noise_types,
            esc50_files=esc50_files,
        )

    total_dataset = {
        "train": build(train_idx, augment=True),
        "validation": build(val_idx),
        "test": build(test_idx),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(total_dataset, f)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
