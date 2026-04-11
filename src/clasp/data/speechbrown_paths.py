"""Parse SpeechBrown-style metadata and resolve audio paths on disk."""

from __future__ import annotations

import json
from pathlib import Path


def audio_relpath(entry: dict) -> str | None:
    """SpeechBrown global_metadata uses `file_path`; other dumps may use `audio_file_path`."""
    path = entry.get("audio_file_path") or entry.get("file_path")
    if path is None:
        return None
    return str(path)


def entry_has_audio_field(entry: dict) -> bool:
    return "audio_file_path" in entry or "file_path" in entry


def load_metadata_entries(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("samples", "data", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
        if data:
            first_val = next(iter(data.values()))
            if isinstance(first_val, dict) and entry_has_audio_field(first_val):
                return list(data.values())
    raise ValueError(
        f"Unsupported JSON structure in {path}: expected a top-level list, "
        "a dict with a list under 'samples', 'data', or 'records', "
        "or a dict whose values are sample objects (dict with 'audio_file_path' or 'file_path')."
    )


def resolve_existing_audio_file(dataset_root: Path, rel: str) -> Path | None:
    """Resolve relative paths from metadata to an existing file on disk.

    SpeechBrown `global_metadata.json` uses `dataset/part1/audios/...`, while
    `dataset_part1.zip` usually unpacks to `dataset_part1/audios/...`. Try both.
    """
    p = Path(rel)
    if p.is_absolute():
        return p.resolve() if p.is_file() else None
    root = dataset_root.resolve()
    candidates = [root / rel]
    if "dataset/part1" in rel:
        candidates.append(root / rel.replace("dataset/part1", "dataset_part1", 1))
    if "dataset_part1" in rel:
        candidates.append(root / rel.replace("dataset_part1", "dataset/part1", 1))
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None
