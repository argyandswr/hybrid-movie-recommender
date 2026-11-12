"""SBERT semantic embedding utilities."""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def encode_semantic_texts(
    model: SentenceTransformer,
    texts: pd.Series,
    batch_size: int = 8,
) -> np.ndarray:
    """Encode non-empty semantic texts into normalized embeddings.

    Parameters
    ----------
    model:
        A loaded SentenceTransformer model.
    texts:
        Non-empty semantic texts, one per row.
    batch_size:
        Number of texts processed per encoding step.

    Returns
    -------
    numpy.ndarray
        A float32 matrix with one normalized embedding per input text.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    if texts.empty:
        raise ValueError("No semantic texts were provided")

    if texts.isna().any():
        raise ValueError("Semantic texts contain missing values")

    if texts.str.strip().eq("").any():
        raise ValueError("Semantic texts contain empty strings")

    embeddings = model.encode(
        texts.tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise RuntimeError(
            "Unexpected embedding shape: "
            f"{embeddings.shape} for {len(texts)} input texts"
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError("Embeddings contain non-finite values")

    return embeddings