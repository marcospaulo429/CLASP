#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.config.settings import get_default_device
from clasp.inference.pipeline import build_final_embeddings, load_model, retrieve_top1


def parse_args():
    parser = argparse.ArgumentParser(description="Run CLASP inference from extracted modules.")
    parser.add_argument("--model-path", required=True, help="Path to trained model .pt")
    parser.add_argument("--audio-embeddings-path", required=True, help="Path to tensor/list of audio embeddings")
    parser.add_argument("--image-embeddings-path", required=True, help="Path to tensor/list of image embeddings")
    parser.add_argument("--text-embeddings-path", required=True, help="Path to tensor/list of text embeddings")
    parser.add_argument("--sample-index", type=int, default=0, help="Query text index")
    return parser.parse_args()


def _load_tensor_or_list(path):
    data = torch.load(path, map_location="cpu")
    if isinstance(data, list):
        return torch.stack(data)
    return data


def main():
    args = parse_args()
    device = get_default_device()

    model = load_model(args.model_path, device)
    audio_embeddings = _load_tensor_or_list(args.audio_embeddings_path).to(device)
    image_embeddings = _load_tensor_or_list(args.image_embeddings_path).to(device)
    text_embeddings = _load_tensor_or_list(args.text_embeddings_path)

    final_embeddings = build_final_embeddings(model, audio_embeddings, image_embeddings)
    idx, score = retrieve_top1(text_embeddings, final_embeddings, args.sample_index, device)

    print(f"Most similar embedding index: {idx}")
    print(f"Cosine similarity: {score:.4f}")


if __name__ == "__main__":
    main()

