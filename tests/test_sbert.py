import numpy as np
import pandas as pd
import pytest

from movie_recommender.features.sbert import encode_semantic_texts


class DummyModel:
    def __init__(self, dim: int = 4):
        self.dim = dim
        self.received = None

    def encode(self, texts, **kwargs):
        self.received = {"texts": texts, **kwargs}
        return np.ones((len(texts), self.dim), dtype=np.float64)


def test_returns_float32_matrix():
    model = DummyModel()
    texts = pd.Series(["first text", "second text"], dtype="string")

    result = encode_semantic_texts(model, texts, batch_size=2)

    assert result.shape == (2, 4)
    assert result.dtype == np.float32


def test_passes_expected_encode_options():
    model = DummyModel()
    texts = pd.Series(["text"], dtype="string")

    encode_semantic_texts(model, texts, batch_size=4)

    assert model.received["batch_size"] == 4
    assert model.received["convert_to_numpy"] is True
    assert model.received["normalize_embeddings"] is True
    assert model.received["texts"] == ["text"]


def test_rejects_empty_series():
    model = DummyModel()

    with pytest.raises(ValueError, match="No semantic texts"):
        encode_semantic_texts(model, pd.Series([], dtype="string"))


def test_rejects_missing_values():
    model = DummyModel()
    texts = pd.Series(["valid", None], dtype="string")

    with pytest.raises(ValueError, match="missing values"):
        encode_semantic_texts(model, texts)


def test_rejects_blank_strings():
    model = DummyModel()
    texts = pd.Series(["valid", "   "], dtype="string")

    with pytest.raises(ValueError, match="empty strings"):
        encode_semantic_texts(model, texts)


def test_rejects_invalid_batch_size():
    model = DummyModel()
    texts = pd.Series(["text"], dtype="string")

    with pytest.raises(ValueError, match="batch_size"):
        encode_semantic_texts(model, texts, batch_size=0)


def test_detects_wrong_output_shape():
    class BadShapeModel:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts) + 1, 4), dtype=np.float32)

    texts = pd.Series(["text"], dtype="string")

    with pytest.raises(RuntimeError, match="Unexpected embedding shape"):
        encode_semantic_texts(BadShapeModel(), texts)


def test_detects_non_finite_values():
    class NaNDummyModel:
        def encode(self, texts, **kwargs):
            result = np.ones((len(texts), 4), dtype=np.float32)
            result[0, 0] = np.nan
            return result

    texts = pd.Series(["text"], dtype="string")

    with pytest.raises(RuntimeError, match="non-finite"):
        encode_semantic_texts(NaNDummyModel(), texts)