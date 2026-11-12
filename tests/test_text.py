import pandas as pd
import pytest

from movie_recommender.preprocessing.text import build_semantic_text


def test_combines_non_empty_fields_with_labels():
    df = pd.DataFrame(
        {
            "overview": ["A thief enters dreams."],
            "tagline": ["Your mind is the scene."],
            "keywords": ["dream, memory"],
        }
    )

    result = build_semantic_text(df)

    assert result.iloc[0] == (
        "Overview: A thief enters dreams.\n\n"
        "Tagline: Your mind is the scene.\n\n"
        "Keywords: dream, memory"
    )


def test_skips_missing_and_whitespace_fields():
    df = pd.DataFrame(
        {
            "overview": ["  A thief   enters dreams. "],
            "tagline": [None],
            "keywords": ["   "],
        }
    )

    result = build_semantic_text(df)

    assert result.iloc[0] == "Overview: A thief enters dreams."


def test_empty_semantic_text_returns_empty_string():
    df = pd.DataFrame(
        {
            "overview": [None],
            "tagline": ["   "],
            "keywords": [None],
        }
    )

    result = build_semantic_text(df)

    assert result.iloc[0] == ""


def test_treats_detected_placeholders_as_empty():
    df = pd.DataFrame(
        {
            "overview": [None],
            "tagline": ["N/a"],
            "keywords": ["  "],
        }
    )

    result = build_semantic_text(df)

    assert result.iloc[0] == ""


def test_preserves_input_index_and_does_not_mutate_dataframe():
    df = pd.DataFrame(
        {
            "overview": ["Movie A", None],
            "tagline": [None, "Tagline B"],
            "keywords": [None, None],
        },
        index=[10, 20],
    )
    before = df.copy(deep=True)

    result = build_semantic_text(df)

    assert result.index.tolist() == [10, 20]
    pd.testing.assert_frame_equal(df, before)


def test_missing_required_column_raises_error():
    df = pd.DataFrame({"overview": ["A movie"]})

    with pytest.raises(ValueError, match="tagline|keywords"):
        build_semantic_text(df)