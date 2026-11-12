"""Movie deduplication utilities."""

import pandas as pd

from movie_recommender.data.validator import RELEVANT_COLUMNS


def deduplicate_movies(
    df: pd.DataFrame,
    id_column: str = "id",
    relevant_columns: tuple[str, ...] = RELEVANT_COLUMNS,
) -> pd.DataFrame:
    """Keep one row per movie ID using metadata completeness.

    Ties are broken by the longest stripped metadata text, then by the
    earliest row in the input.

    All relevant columns are weighted equally. This is a record-selection
    heuristic, not a measure of movie quality.
    """
    if id_column not in df.columns:
        raise KeyError(f"ID column '{id_column}' not found in dataset")

    if df[id_column].isna().any():
        raise ValueError("Dataset contains null movie IDs")

    columns = [column for column in relevant_columns if column in df.columns]
    if not columns:
        raise ValueError("None of the relevant columns are present")

    # Reset index so each row has an unambiguous position: 0, 1, 2, ...
    work = df.reset_index(drop=True)

    # Convert missing values to empty text, then trim whitespace.
    text = work[columns].fillna("").astype(str)
    stripped = text.apply(lambda series: series.str.strip())

    # Number of non-empty relevant fields for each row.
    score = stripped.ne("").sum(axis=1)

    # Tie-break: total character count after trimming whitespace.
    length = stripped.apply(lambda series: series.str.len()).sum(axis=1)

    # Keep the ranking table small instead of sorting all dataset columns.
    ranking = pd.DataFrame(
        {
            id_column: work[id_column].to_numpy(),
            "_score": score.to_numpy(),
            "_length": length.to_numpy(),
            "_order": range(len(work)),
        }
    )

    ranked = ranking.sort_values(
        [id_column, "_score", "_length", "_order"],
        ascending=[True, False, False, True],
    )

    # Save the winning row positions, then restore original row order.
    winners = (
        ranked.drop_duplicates(subset=id_column, keep="first")["_order"]
        .sort_values()
        .to_numpy()
    )

    return work.iloc[winners].reset_index(drop=True)