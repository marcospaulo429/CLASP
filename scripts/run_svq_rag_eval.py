#!/usr/bin/env python3
"""End-to-end RAG eval on SVQ: CLASP retrieval + LLM generation, scored with EM/F1.

Pipeline per spoken query: CLASP encodes the audio -> retrieves the top-k passages
(cosine vs LaBSE passage embeddings) -> a generator LLM (default Qwen3-8B) answers
using those passages -> EM / token-F1 vs the SVQ gold answer span(s). Also reports
retrieval Recall@k so retrieval and generation quality are separable.

Runs on a CUDA box (e.g. a 4090). Use `--dry-run-generator` to exercise the whole
pipeline without loading an LLM (retrieval + scoring only).

Example:
    python scripts/run_svq_rag_eval.py \
        --model-path models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt \
        --config span_reasoning_in_lang --locale en_us \
        --generator Qwen/Qwen3-8B --top-k 5 \
        --output-json artifacts_user/svq_rag_en_us.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from clasp.config.settings import get_default_device
from clasp.rag.generator import DryRunGenerator, HFGenerator
from clasp.rag.svq_rag import ClaspEmbedder, load_svq_reasoning_rows, run_svq_rag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="CLASP fusion checkpoint (.pt)")
    p.add_argument("--config", default="span_reasoning_in_lang",
                   help="SVQ reasoning config (has passage_text + span/spans)")
    p.add_argument("--split", default="test")
    p.add_argument("--locale", default="en_us", help="Locale filter (e.g. en_us); '' for all")
    p.add_argument("--audio-config", default="audio",
                   help="SVQ config holding the waveforms, joined by utt_id (streamed; 'audio' covers all locales)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--top-k", type=int, default=5, help="Passages retrieved and fed to the LLM")
    p.add_argument("--audio-cache-dir", type=Path, default=Path("data/datasets/svq_rag_wav"))
    p.add_argument("--device", default=None)
    # generator
    p.add_argument("--generator", default="Qwen/Qwen3-8B", help="HF model id for the answer LLM")
    p.add_argument("--dry-run-generator", action="store_true",
                   help="Skip the LLM; stub the generator (retrieval + scoring only)")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--enable-thinking", action="store_true", help="Qwen3 thinking mode (default off)")
    p.add_argument("--generator-device", default=None,
                   help="Device for the LLM (default: HF device_map='auto')")
    # output
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--vision-batch-size", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()

    rows = load_svq_reasoning_rows(
        config=args.config, split=args.split,
        locale=(args.locale or None), max_samples=args.max_samples,
        audio_cache_dir=args.audio_cache_dir, audio_config=args.audio_config,
    )
    if not rows:
        raise SystemExit("No SVQ rows loaded (check --config/--locale).")
    print(f"Loaded {len(rows)} SVQ rows (config={args.config}, locale={args.locale or 'all'})")

    embedder = ClaspEmbedder(str(args.model_path), device)

    if args.dry_run_generator:
        generator = DryRunGenerator()
    else:
        generator = HFGenerator(
            model_id=args.generator, device=args.generator_device,
            max_new_tokens=args.max_new_tokens, enable_thinking=args.enable_thinking,
        )

    result = run_svq_rag(embedder, generator, rows, top_k=args.top_k)

    summary = {
        "config": args.config, "locale": args.locale, "n": result.n,
        "top_k": result.top_k, "generator": result.generator,
        "recall_at_k": result.recall_at_k,
        "exact_match": result.exact_match, "f1": result.f1,
    }
    print("\n=== SVQ RAG results ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_row": result.per_row}, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
