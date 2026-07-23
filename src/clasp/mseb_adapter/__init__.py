"""CLASP adapter for the Massive Sound Embedding Benchmark (MSEB).

Exposes CLASP as an ``mseb.encoder.MultiModalEncoder`` so it can be evaluated
on MSEB tasks (SVQ acoustic-hypothesis reranking + SVQ audio->text retrieval).

Importing this package requires the ``[mseb]`` extra, which needs Python >= 3.12.
See ``docs/MSEB.md``.
"""

from clasp.mseb_adapter.clasp_encoder import ClaspMultiModalEncoder

__all__ = ["ClaspMultiModalEncoder"]
