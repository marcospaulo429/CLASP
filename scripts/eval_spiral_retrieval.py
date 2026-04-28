#!/usr/bin/env python3
"""Backward-compatible entrypoint for SPIRAL retrieval; delegates to clasp.evaluation.spiral_runner."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.config.settings import get_default_device
from clasp.evaluation.spiral_runner import run_spiral_retrieval_eval


def main():
    parser = argparse.ArgumentParser(description="CLASP SPIRAL retrieval (wrapper around shared eval stack).")
    parser.add_argument("--data", type=Path, required=True, help="SPIRAL data.jsonl")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "checkpoints" / "CLASP_Concat_Final_Fusion_Encoder.pt",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "spiral")
    parser.add_argument("--audio-base-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size-text", type=int, default=32)
    parser.add_argument(
        "--spiral-chunk-batch-size",
        type=int,
        default=1,
        help="Batch HuBERT/EfficientNet chunk forwards within each wav (long audio).",
    )
    parser.add_argument(
        "--spiral-batch-size-fusion",
        type=int,
        default=32,
        help="Batch size for CLASP fusion encoder.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--hits-k",
        type=str,
        default="1,5,10,50",
        help="Comma-separated K for Hits@K",
    )
    parser.add_argument("--hubert-model", default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--sentence-transformer", default="sentence-transformers/LaBSE")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not args.data.exists():
        print(f"Data not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    dev = torch.device(args.device) if args.device else get_default_device()
    ks = [int(x.strip()) for x in args.hits_k.split(",") if x.strip()]
    run_spiral_retrieval_eval(
        args.data,
        args.model,
        args.output,
        audio_base_dir=args.audio_base_dir,
        extra_search_roots=(ROOT,),
        max_samples=args.max_samples,
        batch_size_text=args.batch_size_text,
        chunk_batch_size_audio=args.spiral_chunk_batch_size,
        batch_size_fusion=args.spiral_batch_size_fusion,
        hubert_model_id=args.hubert_model,
        sentence_model_id=args.sentence_transformer,
        device=dev,
        hits_ks=ks,
    )


if __name__ == "__main__":
    main()
