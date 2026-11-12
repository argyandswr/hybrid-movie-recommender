import pandas as pd
import pytest

from movie_recommender.data.deduplicate import deduplicate_movies

def test_rejects_null_ids():
    df = pd.DataFrame(
        {
            "id": [1, None],
            "overview": ["a", "b"],
        }
    )

    with pytest.raises(ValueError, match="null movie IDs"):
        deduplicate_movies(df)


def test_whitespace_does_not_add_tiebreak_length():
    df = pd.DataFrame(
        {
            "id": [1, 1],
            "title": ["first", "second"],
            "overview": ["same", "same   "],
        }
    )

    result = deduplicate_movies(df)

    # Skor dan panjang setelah strip sama, jadi baris pertama menang.
    assert result.loc[0, "title"] == "first"