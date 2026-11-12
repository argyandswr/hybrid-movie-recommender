"""Text preprocessing for semantic movie features."""

import pandas as pd


SEMANTIC_COLUMNS: tuple[str, ...] = ("overview", "tagline", "keywords")

# These exact values were found in tagline and appear to be placeholders.
PLACEHOLDER_VALUES = {"none", "n/a", "-"}


def _clean_text(value: object) -> str:
    """Convert a value to clean text, treating missing/placeholders as empty."""
    if pd.isna(value):
        return ""

    text = " ".join(str(value).split())

    if text.casefold() in PLACEHOLDER_VALUES:
        return ""

    return text


def build_semantic_text(df: pd.DataFrame) -> pd.Series:
    """Combine overview, tagline, and keywords into one text per movie.

    Non-empty fields are labeled to retain their context. Rows with no
    semantic text return an empty string. The input DataFrame is not modified.
    """
    missing_columns = [
        column for column in SEMANTIC_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing semantic text columns: {missing_columns}")

    semantic_texts: list[str] = []

    for overview, tagline, keywords in zip(
        df["overview"],
        df["tagline"],
        df["keywords"],
        strict=True,
    ):
        fields = (
            ("Overview", overview),
            ("Tagline", tagline),
            ("Keywords", keywords),
        )

        sections = []
        for label, value in fields:
            text = _clean_text(value)
            if text:
                sections.append(f"{label}: {text}")

        semantic_texts.append("\n\n".join(sections))

    return pd.Series(
        semantic_texts,
        index=df.index,
        name="semantic_text",
        dtype="string",
    )