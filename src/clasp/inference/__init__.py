"""Inference utilities for CLASP."""

from .embed_audio import hubert_audio_files
from .pipeline import (
    build_final_embeddings,
    load_model,
    retrieve_top1,
)

