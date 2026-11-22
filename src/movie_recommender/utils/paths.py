"""Project path constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Baseline semantic artifacts: 396.254 films, built before the language filter.
SEMANTIC_BASELINE_ARTIFACTS_DIR = ARTIFACTS_DIR / "semantic_baseline"

# Final semantic artifacts: 388.241 English-only films.
SEMANTIC_ARTIFACTS_DIR = ARTIFACTS_DIR / "semantic"

RESULTS_DIR = PROJECT_ROOT / "results"

MOVIES_PARQUET = DATA_PROCESSED_DIR / "movies.parquet"