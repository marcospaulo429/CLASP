#!/usr/bin/env python3
"""Evaluate a CLASP checkpoint on an MSEB task (SVQ reranking / retrieval).

Wraps CLASP as an ``mseb.encoder.MultiModalEncoder`` and runs one MSEB task via
``mseb.leaderboard.run_benchmark``, printing (and optionally saving) the
per-task ``LeaderboardResult`` JSON (MAP / MRR / WER / CER for reranking).

Requires the ``[mseb]`` extra in a **Python >= 3.12** environment. On this repo's
Apple-Silicon dev machine the SVQ dataset loader (array_record / apache_beam)
segfaults, so real task runs happen on Linux/x86 or via Docker. See ``docs/MSEB.md``.

Examples
--------
    python scripts/run_mseb_task.py \
        --task SVQEnUsQueryReranking \
        --model-path models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt \
        --results-jsonl artifacts_user/svq_rerank_enus.jsonl
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

# Native-lib guards: MSEB pulls TensorFlow + array_record + apache-beam, whose
# native libraries can clash with torch's (protobuf/abseil/CUDA symbol collisions)
# and segfault. Force the pure-Python protobuf impl and quiet TF, then import torch
# FIRST so its libraries load before MSEB's. Override the env vars if they hurt.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import torch  # noqa: F401,E402  (must precede the MSEB task import)

# torchvision pulls torch._dynamo -> triton, whose native (LLVM) libs segfault if
# loaded AFTER TensorFlow/apache-beam/array_record (which the MSEB task import pulls).
# Import it here so triton loads first. If triton itself segfaults on import in this
# env, uninstall it (`uv pip uninstall triton`) — it isn't needed for the eval.
try:
    import torchvision.models  # noqa: F401,E402
except Exception:  # pragma: no cover - torchvision optional at import time
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Importing only the task module we need avoids `from mseb import tasks`, which
# eagerly imports every task family (heavy, and unstable on some platforms).
# Keys are matched as case-insensitive substrings of the task name; order is
# most-specific first so e.g. "documentcrosslang" wins over "document".
_TASK_MODULE_BY_KEYWORD = {
    "reranking": "mseb.tasks.rerankings.query.svq",
    "documentcrosslang": "mseb.tasks.retrievals.document_cross_lang.svq",
    "documentinlang": "mseb.tasks.retrievals.document_in_lang.svq",
    "passagecrosslang": "mseb.tasks.retrievals.passage_cross_lang.svq",
    "passageinlang": "mseb.tasks.retrievals.passage_in_lang.svq",
    "document": "mseb.tasks.retrievals.document_in_lang.svq",
    "passage": "mseb.tasks.retrievals.passage_in_lang.svq",
}


def _guess_task_module(task_name: str) -> str | None:
    key = task_name.lower()
    for keyword, module in _TASK_MODULE_BY_KEYWORD.items():
        if keyword in key:
            return module
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="MSEB task name, e.g. SVQEnUsQueryReranking")
    p.add_argument("--model-path", required=True, help="CLASP fusion checkpoint (.pt)")
    p.add_argument("--results-jsonl", type=Path, default=None, help="Write LeaderboardResult JSON here")
    p.add_argument("--encoder-name", default="clasp", help="Label recorded in the results")
    p.add_argument("--device", default=None, help="torch device (cuda/cpu/mps); default auto")
    p.add_argument("--batch-size", type=int, default=1, help="Encoder batch size")
    p.add_argument("--num-threads", type=int, default=1, help="Runner threads")
    p.add_argument("--cache-dir", default=None, help="Optional runner cache/output dir")
    p.add_argument("--chunk-batch-size", type=int, default=4, help="HuBERT/EfficientNet chunk batch size")
    p.add_argument(
        "--task-module",
        default=None,
        help="Dotted module to import so the task registers (default: guessed from --task).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Register the task class by importing its module (or the whole registry).
    module = args.task_module or _guess_task_module(args.task)
    if module is not None:
        importlib.import_module(module)
    else:
        importlib.import_module("mseb.tasks")  # fall back to full registry

    from mseb import leaderboard
    from mseb import runner as runner_lib
    from mseb.task import get_task_by_name

    from clasp.mseb_adapter.clasp_encoder import ClaspMultiModalEncoder

    encoder = ClaspMultiModalEncoder(
        model_path=args.model_path,
        device=args.device,
        chunk_batch_size=args.chunk_batch_size,
    )
    runner = runner_lib.DirectRunner(
        encoder=encoder,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        output_path=args.cache_dir,
    )
    task = get_task_by_name(args.task)()
    task.setup(runner=runner)

    results = leaderboard.run_benchmark(
        encoder_name=args.encoder_name,
        runner=runner,
        task=task,
        url="https://huggingface.co/llm-lab/CLASP",
        base_model="CLASP",
        tags=["clasp"],
    )

    if args.results_jsonl is not None:
        args.results_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.results_jsonl.open("w", encoding="utf-8") as f:
            for result in results:
                f.write(result.to_json() + "\n")
        print(f"Wrote {len(results)} result(s) to {args.results_jsonl}")

    for result in results:
        print(result.to_json())


if __name__ == "__main__":
    main()
