"""Generate SBERT embeddings for Phase 3."""

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

from movie_recommender.features.sbert import encode_semantic_texts
from movie_recommender.utils.paths import (
    MOVIES_PARQUET,
    RESULTS_DIR,
    SEMANTIC_ARTIFACTS_DIR,
)

MODEL_NAME = "sentence-transformers/multi-qa-distilbert-cos-v1"
BATCH_SIZE = 8
TOKEN_BATCH_SIZE = 2048


def load_inputs() -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the preprocessing artifacts."""
    movies = pd.read_parquet(MOVIES_PARQUET).reset_index(drop=True)
    text_frame = pd.read_parquet(
        SEMANTIC_ARTIFACTS_DIR / "semantic_text.parquet"
    ).reset_index(drop=True)

    if len(movies) != len(text_frame):
        raise ValueError(
            "Row count mismatch: "
            f"movies={len(movies)}, semantic_text={len(text_frame)}"
        )

    if not np.array_equal(
        movies["id"].to_numpy(),
        text_frame["id"].to_numpy(),
    ):
        raise ValueError("Movie IDs are not aligned between artifacts")

    semantic_text = text_frame["semantic_text"].fillna("")
    return movies, semantic_text


def audit_tokens(
    model: SentenceTransformer,
    texts: pd.Series,
) -> pd.Series:
    """Measure token lengths without truncation."""
    tokenizer = model[0].tokenizer
    lengths: list[int] = []

    for start in range(0, len(texts), TOKEN_BATCH_SIZE):
        batch = texts.iloc[start : start + TOKEN_BATCH_SIZE].tolist()
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])

    return pd.Series(
        lengths,
        index=texts.index,
        name="token_length",
    )


def main() -> None:
    movies, semantic_text = load_inputs()

    eligible_mask = semantic_text.str.strip().ne("")
    texts = semantic_text.loc[eligible_mask]
    text_movie_ids = movies.loc[texts.index, "id"].to_numpy(
        dtype=np.int64
    )

    print("movies:", len(movies))
    print("empty semantic text:", int((~eligible_mask).sum()))
    print("texts to encode:", len(texts))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = SentenceTransformer(MODEL_NAME, device=device)
    max_seq_length = model.max_seq_length
    print("model:", MODEL_NAME)
    print("max sequence length:", max_seq_length)

    token_lengths = audit_tokens(model, texts)

    token_length_frame = pd.DataFrame(
        {
            "id": text_movie_ids,
            "token_length": token_lengths.to_numpy(),
        }
    )
    SEMANTIC_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    token_length_frame.to_parquet(
        SEMANTIC_ARTIFACTS_DIR / "token_lengths.parquet",
        index=False,
        compression="zstd",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_movies": len(movies),
        "n_semantic_texts": len(texts),
        "n_empty_semantic_text": int((~eligible_mask).sum()),
        "max_seq_length": int(max_seq_length),
        "n_over_250": int(token_lengths.gt(250).sum()),
        "n_over_512": int(token_lengths.gt(max_seq_length).sum()),
        "token_p50": int(token_lengths.quantile(0.50)),
        "token_p90": int(token_lengths.quantile(0.90)),
        "token_p99": int(token_lengths.quantile(0.99)),
        "token_max": int(token_lengths.max()),
    }
    pd.Series(summary, name="value").to_csv(
        RESULTS_DIR / "semantic_text_audit.csv",
        index_label="metric",
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    embeddings = encode_semantic_texts(
        model=model,
        texts=texts,
        batch_size=BATCH_SIZE,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    np.save(
        SEMANTIC_ARTIFACTS_DIR / "embeddings.npy",
        embeddings,
    )
    np.save(
        SEMANTIC_ARTIFACTS_DIR / "movie_ids.npy",
        text_movie_ids,
    )

    config = {
        "model_name": MODEL_NAME,
        "embedding_dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "n_texts": int(embeddings.shape[0]),
        "n_movies": int(len(movies)),
        "n_empty_semantic_text": int((~eligible_mask).sum()),
        "max_seq_length": int(max_seq_length),
        "batch_size": BATCH_SIZE,
        "device": str(model.device),
        "truncation": "model default (max_seq_length)",
        "n_over_250_tokens": summary["n_over_250"],
        "n_over_512_tokens": summary["n_over_512"],
        "elapsed_seconds": round(elapsed_seconds, 2),
        "peak_gpu_memory_mib": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            if device == "cuda"
            else 0.0
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    (SEMANTIC_ARTIFACTS_DIR / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("embedding shape:", embeddings.shape)
    print("elapsed seconds:", round(elapsed_seconds, 2))
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()