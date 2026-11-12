"""Dataset loading utilities."""

from pathlib import Path 
import pandas as pd

def load_raw_movies(path: str | Path) -> pd.DataFrame:
    """Load the raw movie dataset from a CSV file.

    Parameters
    ----------
    path:
        Path to the raw CSV dataset.

    Returns
    -------
    pandas.DataFrame
        The dataset exactly as stored on disk, without any transformation.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported file format: {path.suffix}. Only CSV files are supported.")

    df = pd.read_csv(path, low_memory=False)

    if df.empty:
        raise ValueError(f"Dataset contains no rows: {path}")

    return df