"""CLASP encoder for the Massive Sound Embedding Benchmark (MSEB).

``ClaspMultiModalEncoder`` wraps CLASP behind MSEB's ``MultiModalEncoder``
interface. It encodes:

* ``types.Sound``  -> CLASP fused audio embedding (HuBERT + EfficientNet-B7
  spectrogram-image, fused by the trained ``HubertLabseConcat`` head), 768-d.
* ``types.Text``   -> LaBSE sentence embedding, 768-d.

Both are L2-normalized so MSEB's default ``dot_product`` distance equals the
cosine similarity CLASP was trained with. This is exactly CLASP's cross-modal
retrieval setup, so the same encoder drives both the SVQ acoustic-hypothesis
reranking task and SVQ audio->text retrieval.

Requires the ``[mseb]`` extra (Python >= 3.12). See ``docs/MSEB.md``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from mseb import encoder as mseb_encoder
from mseb import types

from clasp.inference.embed_audio import hubert_numpy_waveform
from clasp.inference.pipeline import load_model
from clasp.inference.spectrogram_image import (
    efficientnet_embedding_from_waveform,
    load_efficientnet_b7,
)

HUBERT_MODEL = "facebook/hubert-large-ls960-ft"
LABSE_MODEL = "sentence-transformers/LaBSE"
TARGET_SR = 16_000


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; safe against zero vectors."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


class ClaspMultiModalEncoder(mseb_encoder.MultiModalEncoder):
    """MSEB encoder backed by a trained CLASP fusion checkpoint + LaBSE."""

    def __init__(
        self,
        model_path,
        device: str | None = None,
        hubert_model: str = HUBERT_MODEL,
        labse_model: str = LABSE_MODEL,
        chunk_samples: int = 320_000,
        chunk_batch_size: int = 4,
    ):
        super().__init__()
        self.model_path = str(model_path)
        self._device_str = device
        self.hubert_model_name = hubert_model
        self.labse_model_name = labse_model
        self.chunk_samples = chunk_samples
        self.chunk_batch_size = chunk_batch_size
        # Heavy handles are created lazily in _setup().
        self.device: torch.device | None = None
        self.clasp = None
        self.hubert_processor = None
        self.hubert = None
        self.effnet = None
        self.effnet_preprocess = None
        self.labse = None

    # -- lifecycle -----------------------------------------------------------
    def _resolve_device(self) -> torch.device:
        if self._device_str:
            return torch.device(self._device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _setup(self):
        from sentence_transformers import SentenceTransformer
        from transformers import AutoProcessor, HubertModel

        self.device = self._resolve_device()
        self.clasp = load_model(self.model_path, self.device)  # returns .eval() model
        self.hubert_processor = AutoProcessor.from_pretrained(self.hubert_model_name)
        self.hubert = HubertModel.from_pretrained(self.hubert_model_name).to(self.device)
        self.hubert.eval()
        self.effnet, self.effnet_preprocess = load_efficientnet_b7(self.device)
        self.labse = SentenceTransformer(self.labse_model_name, device=str(self.device))

    def _check_input_types(self, batch: Sequence[types.MultiModalObject]) -> None:
        for x in batch:
            if not isinstance(x, (types.Sound, types.Text)):
                raise ValueError(
                    "ClaspMultiModalEncoder supports Sound and Text inputs, "
                    f"got {type(x).__name__}."
                )

    # -- encoding ------------------------------------------------------------
    def _sound_to_waveform(self, sound: types.Sound) -> np.ndarray:
        """Mono 16 kHz float32 waveform, honoring an explicit [start, end] segment."""
        resampled = mseb_encoder.resample_sound(sound, TARGET_SR, np.float32)
        y = np.asarray(resampled.waveform, dtype=np.float32).reshape(-1)
        ctx = resampled.context
        start_s = float(getattr(ctx, "waveform_start_second", 0.0) or 0.0)
        end_s = float(getattr(ctx, "waveform_end_second", np.finfo(np.float32).max))
        if np.isfinite(end_s) and end_s < np.finfo(np.float32).max:
            a = max(0, int(start_s * TARGET_SR))
            b = int(end_s * TARGET_SR)
            if 0 <= a < b <= y.size:
                y = y[a:b]
        return y

    def _encode_sound(self, sound: types.Sound) -> types.SoundEmbedding:
        y = self._sound_to_waveform(sound)
        hub = hubert_numpy_waveform(
            y,
            self.hubert_processor,
            self.hubert,
            self.device,
            chunk_samples=self.chunk_samples,
            chunk_batch_size=self.chunk_batch_size,
            pooling="mean",
        ).reshape(1, -1).to(self.device)
        img = efficientnet_embedding_from_waveform(
            y,
            self.effnet,
            self.effnet_preprocess,
            self.device,
            chunk_samples=self.chunk_samples,
            chunk_batch_size=self.chunk_batch_size,
            pooling="mean",
        ).reshape(1, -1).to(self.device)
        with torch.no_grad():
            fused = self.clasp(hub, img)
        embedding = _l2_normalize(fused.detach().cpu().numpy())  # [1, 768]
        return types.SoundEmbedding(
            embedding=embedding,
            timestamps=np.array(
                [[sound.context.waveform_start_second, sound.context.waveform_end_second]],
                dtype=np.float32,
            ),
            context=sound.context,
        )

    def _encode_text(self, text_obj: types.Text) -> types.TextEmbedding:
        vec = self.labse.encode([text_obj.text], convert_to_numpy=True)
        embedding = _l2_normalize(np.asarray(vec, dtype=np.float32))  # [1, 768]
        return types.TextEmbedding(
            embedding=embedding,
            spans=np.array([[0, len(text_obj.text)]], dtype=np.int64),
            context=text_obj.context,
        )

    def _encode(
        self, batch: Sequence[types.MultiModalObject]
    ) -> Sequence[types.MultiModalEmbedding]:
        out: list[types.MultiModalEmbedding] = []
        for x in batch:
            if isinstance(x, types.Sound):
                out.append(self._encode_sound(x))
            elif isinstance(x, types.Text):
                out.append(self._encode_text(x))
            else:  # pragma: no cover - guarded by _check_input_types
                raise ValueError(f"Unsupported input type: {type(x).__name__}")
        return out
