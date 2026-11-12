from movie_recommender.utils.paths import MOVIES_PARQUET, PROJECT_ROOT


def test_project_root_contains_pyproject():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_movies_parquet_path_is_absolute():
    assert MOVIES_PARQUET.is_absolute()