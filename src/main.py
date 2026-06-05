"""Top-level driver: runs every analysis step in order."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dispersion.main import run as run_dispersion  # noqa: E402
from preprocessing.main import run as run_preprocessing  # noqa: E402


def main():
    run_preprocessing()
    run_dispersion()


if __name__ == "__main__":
    main()
