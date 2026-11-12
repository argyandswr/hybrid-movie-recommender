"""Preprocess raw movie data into Phase 3 artifacts."""

import pandas as pd

from movie_recommender.data.deduplicate import deduplicate_movies
from movie_recommender.data.loader import load_raw_movies
from movie_recommender.data.validator import (
    RELEVANT_COLUMNS,
    require_columns,
    validate_movies,
)
from movie_recommender.preprocessing.text import build_semantic_text
from movie_recommender.utils.paths import (
    DATA_RAW_DIR,
    MOVIES_PARQUET,
    SEMANTIC_ARTIFACTS_DIR,
)

RAW_PATH = DATA_RAW_DIR / "TMDB  IMDB Movies Dataset.csv"


def main() -> None:
    movies = load_raw_movies(RAW_PATH)

    report = validate_movies(movies)
    print("raw rows:", report["n_rows"])
    print("raw columns:", report["n_columns"])
    print("duplicate id groups:", report["n_duplicate_ids"])
    print("null ids:", report["n_null_ids"])

    require_columns(movies, RELEVANT_COLUMNS + ("id",))

    movies_dedup = deduplicate_movies(movies)
    print("deduplicated rows:", len(movies_dedup))
    print("unique ids:", movies_dedup["id"].is_unique)

    MOVIES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    movies_dedup.to_parquet(
        MOVIES_PARQUET,
        index=False,
        compression="zstd",
    )

    semantic_text = build_semantic_text(movies_dedup)

    semantic_text_frame = pd.DataFrame(
        {
            "id": movies_dedup["id"],
            "semantic_text": semantic_text,
        }
    )

    SEMANTIC_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    semantic_text_path = SEMANTIC_ARTIFACTS_DIR / "semantic_text.parquet"
    semantic_text_frame.to_parquet(
        semantic_text_path,
        index=False,
        compression="zstd",
    )

    print("semantic text rows:", len(semantic_text_frame))
    print("empty semantic text:", int(semantic_text.eq("").sum()))
    print("saved:", MOVIES_PARQUET)
    print("saved:", semantic_text_path)


if __name__ == "__main__":
    main()