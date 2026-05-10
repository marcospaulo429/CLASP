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
    evaluate_model_on_paragraph_groups,
)
from clasp.evaluation.ranking_metrics import compute_ranking_metrics, similarity_matrix_to_rows
from clasp.evaluation.retrieval_plots import save_retrieval_plot
from clasp.evaluation.spiral_runner import run_spiral_retrieval_eval
from clasp.inference.pipeline import load_model
from clasp.retrieval.search import build_similarity_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Run CLASP retrieval evaluation.")
    parser.add_argument(
        "--mode",
        choices=["candidate", "matrix", "spiral", "paragraph_grouped"],
        default="candidate",
        help=(
            "spiral: JSONL + on-the-fly embeddings. "
            "candidate/matrix: pickle dataset (1 amostra por linha). "
            "paragraph_grouped: pickle 'chunked' c/ paragraph_id; max-sim por par\u00e1grafo."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Pickle total_dataset path (candidate/matrix) or SPIRAL data.jsonl (spiral)",
    )
    parser.add_argument("--model-path", help="Required for candidate and spiral modes")
    parser.add_argument("--audio-key", default="hubert-emb")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--emb-key", default="clasp_emb", help="Embedding key for matrix mode")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--by-source", action="store_true")
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="If set (matrix mode only), save retrieval metrics plot as PNG to this path.",
    )
    parser.add_argument(
        "--retrieval-plot-dir",
        type=Path,
        default=None,
        help="If set (matrix mode), save retrieval_summary.png under this directory.",
    )
    parser.add_argument(
        "--hits-k",
        type=str,
        default="1,5,10,50",
        help="Comma-separated K values for Hits@K in ranking metrics and plot.",
    )
    # SPIRAL-only
    parser.add_argument(
        "--spiral-audio-base",
        type=Path,
        default=None,
        help="Base directory to resolve SPIRAL wav paths (default: parent of JSONL)",
    )
    parser.add_argument(
        "--spiral-output-dir",
        type=Path,
        default=ROOT / "results" / "spiral",
        help="Output directory for SPIRAL plots and JSON",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="SPIRAL: cap number of samples",
    )
    parser.add_argument(
        "--batch-size-text",
        type=int,
        default=32,
        help="SPIRAL: SentenceTransformer encode batch size",
    )
    parser.add_argument(
        "--spiral-chunk-batch-size",
        type=int,
        default=1,
        help=(
            "SPIRAL: batch multiple same-file time chunks (HuBERT + EfficientNet) per forward. "
            "Helps long wavs; short clips (single chunk) see little change."
        ),
    )
    parser.add_argument(
        "--spiral-batch-size-fusion",
        type=int,
        default=32,
        help="SPIRAL: batch size for CLASP fusion encoder over rows of HuBERT+spec vectors.",
    )
    parser.add_argument("--hubert-model", default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--sentence-transformer", default="sentence-transformers/LaBSE")
    parser.add_argument(
        "--spiral-audio-pooling",
        type=str,
        default="mean",
        choices=["mean", "max_sim"],
        help="SPIRAL: mean = global mean over time chunks; max_sim = ColBERT-style max over chunks.",
    )
    parser.add_argument(
        "--spiral-chunk-samples",
        type=int,
        default=320_000,
        help="SPIRAL: window size in samples (16 kHz) for HuBERT/EfficientNet and timestamp→chunk map.",
    )
    return parser.parse_args()


def _parse_hits_k(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    args = parse_args()
    device = get_default_device()

    if args.mode == "spiral":
        if not args.model_path:
            raise ValueError("--model-path is required for spiral mode")
        dp = Path(args.dataset_path)
        if not dp.is_file():
            raise FileNotFoundError(f"SPIRAL JSONL not found: {dp}")
        run_spiral_retrieval_eval(
            dp,
            Path(args.model_path),
            Path(args.spiral_output_dir),
            audio_base_dir=args.spiral_audio_base,
            extra_search_roots=(ROOT,),
            max_samples=args.max_samples,
            batch_size_text=args.batch_size_text,
            chunk_batch_size_audio=args.spiral_chunk_batch_size,
            batch_size_fusion=args.spiral_batch_size_fusion,
            hubert_model_id=args.hubert_model,
            sentence_model_id=args.sentence_transformer,
            device=device,
            hits_ks=_parse_hits_k(args.hits_k),
            chunk_samples=args.spiral_chunk_samples,
            audio_pooling=args.spiral_audio_pooling,
        )
        return

    with open(args.dataset_path, "rb") as f:
        total_dataset = pickle.load(f)

    # support PKLs that use 'validation' instead of 'test'
    if "test" not in total_dataset and "validation" in total_dataset:
        total_dataset["test"] = total_dataset["validation"]

    if args.mode == "candidate":
        if args.plot_out is not None:
            print(
                "Warning: --plot-out is ignored in candidate mode (full similarity matrix required).",
                file=sys.stderr,
            )
        if args.retrieval_plot_dir is not None:
            print(
                "Warning: --retrieval-plot-dir is ignored in candidate mode.",
                file=sys.stderr,
            )
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

    if args.mode == "paragraph_grouped":
        if not args.model_path:
            raise ValueError("--model-path is required for paragraph_grouped mode")
        if "paragraph_id" not in total_dataset["test"]:
            raise KeyError(
                "PKL 'test/validation' split does not contain 'paragraph_id'. "
                "Re-build with --pooling-mode chunked."
            )
        model = load_model(args.model_path, device)
        ks = _parse_hits_k(args.hits_k)
        results = evaluate_model_on_paragraph_groups(
            model,
            total_dataset["test"],
            device,
            audio_key=args.audio_key,
            text_key=args.text_key,
            ks=ks,
            batch_size=max(1, args.batch_size * 16),
        )
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

    ks = _parse_hits_k(args.hits_k)
    rows = similarity_matrix_to_rows(similarity_matrix)
    ranking_metrics, ranks = compute_ranking_metrics(rows, ks=ks)
    print("Ranking metrics:", ranking_metrics)

    if args.plot_out is not None:
        save_retrieval_plot(ranking_metrics, ranks, args.plot_out, ks=ks)

    if args.retrieval_plot_dir is not None:
        out_dir = Path(args.retrieval_plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_retrieval_plot(
            ranking_metrics,
            ranks,
            out_dir / "retrieval_summary.png",
            ks=ks,
            title="Matrix retrieval",
            hits_display="percent",
        )


if __name__ == "__main__":
    main()
