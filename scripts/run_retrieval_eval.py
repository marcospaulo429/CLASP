#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.config.settings import get_default_device
from clasp.data.datasets import TestDataset, build_test_metadata
from clasp.evaluation.metrics import (
    evaluate_matrix,
    evaluate_matrix_by_source,
    evaluate_model_on_candidates,
)
from clasp.inference.pipeline import load_model
from clasp.retrieval.search import build_similarity_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Run CLASP retrieval evaluation.")
    parser.add_argument("--dataset-path", required=True, help="Path to total_dataset .pkl")
    parser.add_argument("--mode", choices=["candidate", "matrix"], default="candidate")
    parser.add_argument("--model-path", help="Required for candidate mode")
    parser.add_argument("--audio-key", default="hubert-emb")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--emb-key", default="clasp_emb", help="Embedding key for matrix mode")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--by-source", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_default_device()

    with open(args.dataset_path, "rb") as f:
        total_dataset = pickle.load(f)

    if args.mode == "candidate":
        if not args.model_path:
            raise ValueError("--model-path is required for candidate mode")

        test_len_data = len(total_dataset["test"][args.text_key])
        test_metadata = build_test_metadata(test_len_data, args.num_candidates)
        test_dataset = TestDataset(total_dataset["test"], test_metadata, args.audio_key, args.text_key)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
        model = load_model(args.model_path, device)
        results = evaluate_model_on_candidates(model, test_loader, device, threshold=args.threshold)
        print(results)
        return

    query_embeddings = total_dataset["test"][args.emb_key]
    candidate_embeddings = total_dataset["test"][args.text_key]
    similarity_matrix = build_similarity_matrix(query_embeddings, candidate_embeddings)

    if args.by_source:
        results = evaluate_matrix_by_source(
            similarity_matrix,
            total_dataset["test"]["source"],
            threshold=args.threshold,
        )
    else:
        results = evaluate_matrix(similarity_matrix, threshold=args.threshold)
    print(results)


if __name__ == "__main__":
    main()

