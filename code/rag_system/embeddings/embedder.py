from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_system.utils.device import get_torch_device


class SentenceEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = get_torch_device()
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.astype("float32")
