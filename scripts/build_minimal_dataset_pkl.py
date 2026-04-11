#!/usr/bin/env python3
"""Pickle sintético mínimo para testar run_retrieval_eval --mode candidate."""
import argparse
import pickle
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.inference.pipeline import register_pickled_fusion_classes_for_torch_load

def infer_dims(model):
    audio_in = model.audio_seq[0].in_features
    image_in = model.image_seq[0].in_features
    return audio_in, image_in

def make_split(n, audio_dim, text_dim, image_dim, gen):
    return {
        "hubert-emb": [torch.randn(audio_dim, generator=gen) for _ in range(n)],
        "text": [torch.randn(text_dim, generator=gen) for _ in range(n)],
        "image": [torch.randn(image_dim, generator=gen) for _ in range(n)],
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--output", default="data/datasets/total_dataset_minimal.pkl")
    p.add_argument("--text-dim", type=int, default=768)
    p.add_argument("--test-samples", type=int, default=128)
    p.add_argument("--train-samples", type=int, default=32)
    p.add_argument("--val-samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.test_samples < 100:
        args.test_samples = 100  # precisa para num-candidates 100

    random.seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    register_pickled_fusion_classes_for_torch_load()
    try:
        model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(args.model_path, map_location="cpu")
    adim, idim = infer_dims(model)
    del model

    data = {
        "train": make_split(args.train_samples, adim, args.text_dim, idim, gen),
        "validation": make_split(args.val_samples, adim, args.text_dim, idim, gen),
        "test": make_split(args.test_samples, adim, args.text_dim, idim, gen),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(data, f)

    print(f"OK: {out} (audio={adim}, text={args.text_dim}, image={idim}); embeddings aleatórios.")

if __name__ == "__main__":
    main()