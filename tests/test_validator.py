import pandas as pd
import pytest

from movie_recommender.data.validator import require_columns, validate_movies


def make_movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 1, 2, None],
            "title": ["A", "A", "B", "C"],
            "overview": ["foo", "   ", None, "bar"],
            "genres": ["Drama", "Drama", "Comedy", "Comedy"],
        }
    )


def test_reports_basic_counts():
    report = validate_movies(make_movies())

    assert report["n_rows"] == 4
    assert report["n_columns"] == 4
    assert report["n_unique_ids"] == 2
    assert report["n_null_ids"] == 1


def test_detects_duplicate_ids():
    report = validate_movies(make_movies())

    assert report["n_duplicate_ids"] == 1
    assert report["n_duplicate_id_rows"] == 2
    assert report["n_duplicate_rows"] == 0


def test_separates_missing_from_empty_string():
    report = validate_movies(make_movies())
    stats = report["relevant_stats"]["overview"]

    assert stats["n_missing"] == 1
    assert stats["n_empty"] == 1


def test_reports_absent_relevant_columns():
    report = validate_movies(make_movies())

    assert set(report["absent_relevant"]) == {
        "tagline",
        "keywords",
        "directors",
        "writers",
        "cast",
    }


def test_does_not_mutate_input():
    df = make_movies()
    before = df.copy(deep=True)

    validate_movies(df)

    pd.testing.assert_frame_equal(df, before)


def test_requires_id_column():
    with pytest.raises(KeyError):
        validate_movies(pd.DataFrame({"title": ["A"]}))


def test_require_columns_raises():
    with pytest.raises(ValueError, match="tagline"):
        require_columns(make_movies(), ["id", "tagline"])


def test_require_columns_passes():
    require_columns(make_movies(), ["id", "overview"])