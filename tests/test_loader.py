import pandas as pd
import pytest

from movie_recommender.data.loader import load_raw_movies


def test_loads_valid_csv(tmp_path):
    path = tmp_path / "movies.csv"
    path.write_text("id,title\n1,A\n", encoding="utf-8")

    df = load_raw_movies(path)

    assert df.shape == (1, 2)
    assert df.loc[0, "title"] == "A"


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_movies(tmp_path / "nope.csv")


def test_directory_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_movies(tmp_path)


def test_wrong_extension(tmp_path):
    path = tmp_path / "movies.xlsx"
    path.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_raw_movies(path)


def test_header_only_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("id,title\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no rows"):
        load_raw_movies(path)


def test_physically_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(pd.errors.EmptyDataError):
        load_raw_movies(path)