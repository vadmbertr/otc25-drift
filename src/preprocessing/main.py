"""End-to-end preprocessing pipeline."""
import argparse
import sys
from pathlib import Path

# Make ../utils importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing import figures, load, qc, smoothing, stats  # noqa: E402
from utils.io_paths import DATA_PREPROCESSED, DRIFTERS_PARQUET  # noqa: E402


def run(skip_figures: bool = False) -> None:
    raw = load.load_all()
    print(f"[1] loaded {len(raw)} raw rows ({raw['drifter_id'].nunique()} drifters)")

    qc_df = qc.run_qc(raw)
    print(
        f"[2] qc: {len(qc_df)} rows, "
        f"{qc_df['segment_id'].nunique()} segments, "
        f"{qc_df['drifter_id'].nunique()} drifters"
    )

    smooth = smoothing.smooth_all(qc_df)
    print(f"[3] smoothed: {len(smooth)} hourly rows, {smooth['segment_id'].nunique()} segments")

    DATA_PREPROCESSED.mkdir(parents=True, exist_ok=True)
    smooth.to_parquet(DRIFTERS_PARQUET, index=False)
    print(f"[4] wrote {DRIFTERS_PARQUET}")

    if not skip_figures:
        figures.make_all(raw, qc_df, smooth)
        print("[5] figures written")

    stats.make_all(raw, smooth)
    print("[6] stats tables written")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-figures", action="store_true")
    args = p.parse_args()
    run(skip_figures=args.skip_figures)


if __name__ == "__main__":
    main()
