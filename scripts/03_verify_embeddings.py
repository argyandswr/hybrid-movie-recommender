"""Verify semantic embedding artifacts."""

import json

import numpy as np

from movie_recommender.utils.paths import SEMANTIC_ARTIFACTS_DIR

SPOT_CHECK_IDS = (27205, 157336, 155, 862)


def main() -> None:
    embeddings_path = SEMANTIC_ARTIFACTS_DIR / "embeddings.npy"
    ids_path = SEMANTIC_ARTIFACTS_DIR / "movie_ids.npy"
    config_path = SEMANTIC_ARTIFACTS_DIR / "config.json"

    embeddings = np.load(embeddings_path, mmap_mode="r")
    movie_ids = np.load(ids_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    print("embedding shape:", embeddings.shape)
    print("embedding dtype:", embeddings.dtype)
    print("id shape:", movie_ids.shape)
    print("id dtype:", movie_ids.dtype)
    print("unique ids:", len(np.unique(movie_ids)))

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix")

    if embeddings.shape[0] != len(movie_ids):
        raise ValueError(
            "Embedding rows and movie IDs are not aligned: "
            f"{embeddings.shape[0]} vs {len(movie_ids)}"
        )

    if config["embedding_dim"] != embeddings.shape[1]:
        raise ValueError(
            "Embedding dimension does not match config.json: "
            f"{embeddings.shape[1]} vs {config['embedding_dim']}"
        )

    if config["n_texts"] != embeddings.shape[0]:
        raise ValueError(
            "n_texts in config.json does not match embedding rows"
        )

    sample_norms = np.linalg.norm(embeddings[:1000], axis=1)
    print("sample norm min:", sample_norms.min())
    print("sample norm max:", sample_norms.max())
    print("finite sample:", np.isfinite(embeddings[:1000]).all())

    print("config model:", config["model_name"])
    print("config n_texts:", config["n_texts"])
    print("config max_seq_length:", config["max_seq_length"])

    for movie_id in SPOT_CHECK_IDS:
        rows = np.where(movie_ids == movie_id)[0]
        print(f"movie id {movie_id}: rows={rows}")


if __name__ == "__main__":
    main()