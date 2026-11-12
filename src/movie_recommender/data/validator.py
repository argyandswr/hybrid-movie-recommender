"""Dataset validation utilities."""

from collections.abc import Iterable, Sequence

import pandas as pd

RELEVANT_COLUMNS: tuple[str, ...] = (
    "overview",
    "tagline",
    "keywords",
    "genres",
    "directors",
    "writers",
    "cast",
)


def _count_blank(series: pd.Series) -> int:
    """Count values that are empty or whitespace-only strings."""
    if not pd.api.types.is_string_dtype(series.dtype):
        return 0
    return int(series.dropna().astype(str).str.strip().eq("").sum())


def validate_movies(
    df: pd.DataFrame,
    id_column: str = "id",
    relevant_columns: Iterable[str] = RELEVANT_COLUMNS,
) -> dict[str, object]:
    """Inspect the dataset and return a validation report.

    The function only observes the data; it never mutates or filters it.
    """
    if id_column not in df.columns:
        raise KeyError(f"ID column '{id_column}' not found in dataset")

    relevant_columns = tuple(relevant_columns)
    n_rows = len(df)

    missing = pd.DataFrame(index=df.columns)
    missing["n_missing"] = df.isna().sum()
    missing["n_empty"] = [_count_blank(df[column]) for column in df.columns]
    missing["n_blank"] = missing["n_missing"] + missing["n_empty"]
    missing["pct_blank"] = (
        (missing["n_blank"] / n_rows * 100).round(2) if n_rows else 0.0
    )
    missing = missing[missing["n_blank"] > 0]

    duplicate_id_rows = df[df[id_column].duplicated(keep=False)]

    report: dict[str, object] = {
        "n_rows": int(n_rows),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str),
        "missing": missing,
        "n_duplicate_rows": int(df.duplicated().sum()),
        "n_duplicate_id_rows": int(len(duplicate_id_rows)),
        "n_duplicate_ids": int(duplicate_id_rows[id_column].nunique()),
        "n_unique_ids": int(df[id_column].nunique()),
        "n_null_ids": int(df[id_column].isna().sum()),
        "relevant_stats": {
            column: {
                "n_missing": int(df[column].isna().sum()),
                "n_empty": _count_blank(df[column]),
            }
            for column in relevant_columns
            if column in df.columns
        },
        "absent_relevant": [
            column for column in relevant_columns if column not in df.columns
        ],
    }

    return report


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")