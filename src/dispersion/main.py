"""Run the dispersion analysis: pairs -> D^2 -> inclusion tests -> plots."""
import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dispersion import figures, geom_test, pairs as pair_mod, regimes, stats  # noqa: E402
from utils.io_paths import (  # noqa: E402
    DATA_DISPERSION_DIR,
    DISPERSION_PAIRS_PARQUET,
    DRIFTERS_PARQUET,
)


def run(skip_figures: bool = False) -> None:
    drifters = pd.read_parquet(DRIFTERS_PARQUET)
    print(f"[1] loaded {len(drifters)} smoothed rows ({drifters['drifter_id'].nunique()} drifters)")

    cluster_pairs = pair_mod.build_pairs(drifters)
    n_pairs = cluster_pairs["pair_id"].nunique() if not cluster_pairs.empty else 0
    print(
        f"[2] cluster pairs: {n_pairs} pairs across "
        f"{cluster_pairs['cluster'].nunique() if not cluster_pairs.empty else 0} clusters"
    )

    DATA_DISPERSION_DIR.mkdir(parents=True, exist_ok=True)
    if not cluster_pairs.empty:
        cluster_pairs.to_parquet(DISPERSION_PAIRS_PARQUET, index=False)
        print(f"[3] wrote {DISPERSION_PAIRS_PARQUET}")

    comparisons = {}
    if not cluster_pairs.empty:
        for name, focus in (("marget", "S-MARGET"), ("melodi", "S-MELODI"),
                            ("omb", "S-OMB")):
            cmp = geom_test.compare(cluster_pairs, focus)
            if cmp is None:
                print(f"[4] {focus} test skipped (no {focus} or SS pairs)")
                continue
            comparisons[name] = cmp
            print(
                f"[4] {focus} test: SS pairs max n = {int(cmp.n_base.max())}, "
                f"{focus} pairs max n = {int(cmp.n_focus.max())}"
            )
            drift = "exact" if cmp.drifter_exact else f"{cmp.drifter_n_perm} samp"
            for lag in cmp.perm_lags_h:
                idx = int(np.argmin(np.abs(cmp.lag_h - lag)))
                print(f"      @ {lag/24:.1f} d: drifter-perm p={cmp.drifter_p[idx]:.4f} ({drift})")

    ss = cluster_pairs[cluster_pairs["type_pair"] == "SS"] if not cluster_pairs.empty \
        else cluster_pairs
    fits = regimes.compute_all(ss) if not ss.empty else {}
    if fits:
        lam = fits["pooled"].lambda_per_day
        print(f"[5] regime fits: pooled lambda = {lam:.3f}/day "
              f"(e-fold {1/lam:.1f} d), {len(fits)-1} clusters")

    if not skip_figures:
        figures.make_all(cluster_pairs, comparisons, drifters, fits=fits)
        print("[6] figures written")
    stats.make_all(cluster_pairs, comparisons, fits=fits)
    print("[7] stats tables written")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-figures", action="store_true")
    args = p.parse_args()
    run(skip_figures=args.skip_figures)


if __name__ == "__main__":
    main()
