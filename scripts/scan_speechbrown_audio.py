#!/usr/bin/env python3
"""List SpeechBrown metadata rows whose audio is missing, too short, or unreadable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.data.speechbrown_paths import (  # noqa: E402
    audio_relpath,
    load_metadata_entries,
    resolve_existing_audio_file,
)
from clasp.inference.audio_preprocess import audio_duration_seconds  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata-json", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument(
        "--min-audio-seconds",
        type=float,
        default=0.5,
        help="Report rows strictly below this duration (header-based; fast).",
    )
    p.add_argument("--output-bad", type=Path, default=None, help="Write JSON list of bad rows.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_metadata_entries(args.metadata_json)
    bad: list[dict] = []
    ok = 0
    for e in entries:
        rel = audio_relpath(e)
        if rel is None:
            bad.append({"reason": "no_file_path", "entry_id": e.get("id")})
            continue
        abs_p = resolve_existing_audio_file(args.dataset_root, rel)
        if abs_p is None:
            bad.append({"reason": "file_not_found", "rel": rel, "entry_id": e.get("id")})
            continue
        try:
            dur = audio_duration_seconds(abs_p)
        except OSError as ex:
            bad.append({"reason": f"read_error:{ex}", "path": str(abs_p), "entry_id": e.get("id")})
            continue
        if dur < args.min_audio_seconds:
            bad.append(
                {
                    "reason": "too_short",
                    "seconds": dur,
                    "path": str(abs_p),
                    "entry_id": e.get("id"),
                }
            )
        else:
            ok += 1

    print(f"OK (>= {args.min_audio_seconds}s): {ok}")
    print(f"Bad: {len(bad)}")
    if args.output_bad:
        args.output_bad.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_bad, "w", encoding="utf-8") as f:
            json.dump(bad, f, indent=2)
        print(f"Wrote {args.output_bad}")


if __name__ == "__main__":
    main()
