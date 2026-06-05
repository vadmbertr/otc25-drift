from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_RAW_MARGET = DATA_RAW / "marget"
DATA_RAW_SPOT = DATA_RAW / "spot"
DATA_RAW_MELODI = DATA_RAW / "melodi"
DATA_RAW_OMB = DATA_RAW / "omb"
DEPLOY_TIME_CSV = DATA_RAW / "deploy_time.csv"

DATA_PREPROCESSED = REPO_ROOT / "data" / "preprocessed"
DRIFTERS_PARQUET = DATA_PREPROCESSED / "drifters.parquet"
STATS_DIR = DATA_PREPROCESSED / "stats"

PLOTS_DIR = REPO_ROOT / "plots" / "preprocessing"

DATA_DISPERSION_DIR = DATA_PREPROCESSED.parent / "dispersion"
DISPERSION_PAIRS_PARQUET = DATA_DISPERSION_DIR / "pairs.parquet"
DISPERSION_STATS_DIR = DATA_DISPERSION_DIR / "stats"
PLOTS_DISPERSION_DIR = REPO_ROOT / "plots" / "dispersion"

DATA_COLOCATED_DIR = DATA_PREPROCESSED.parent / "colocated"
DRIFTERS_COLOCATED_PARQUET = DATA_COLOCATED_DIR / "drifters_colocated.parquet"
PLOTS_COLOCATION_DIR = REPO_ROOT / "plots" / "colocation"
