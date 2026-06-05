"""LaTeX summary tables for the dispersion analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.io_paths import DISPERSION_STATS_DIR

from .geom_test import InclusionComparison
from .regimes import RegimeFit


def _ensure_dir():
    DISPERSION_STATS_DIR.mkdir(parents=True, exist_ok=True)


def pair_counts(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster unique-pair counts, one column per ``type_pair`` label."""
    labels = sorted(pairs["type_pair"].unique())
    rows = []
    for cluster_id, sub in pairs.groupby("cluster", sort=True):
        counts = sub.groupby("type_pair")["pair_id"].nunique().to_dict()
        row = {"cluster": cluster_id}
        for lab in labels:
            row[f"{lab}_pairs"] = counts.get(lab, 0)
        row["total_pairs"] = sub["pair_id"].nunique()
        row["max_lag_days"] = float(sub["lag_h"].max()) / 24.0
        rows.append(row)
    return pd.DataFrame(rows)


def inclusion_summary(cmp: InclusionComparison) -> pd.DataFrame:
    """At each reference lag, report base/focus means, the baseline bootstrap CI
    envelope, and the drifter-level permutation p-value."""
    if cmp is None:
        return pd.DataFrame()
    rows = []
    for lag in cmp.perm_lags_h:
        idx = int(np.argmin(np.abs(cmp.lag_h - lag)))
        rows.append(
            {
                "lag_days": float(lag) / 24.0,
                "D2_SS_km2": float(cmp.d2_base[idx]),
                f"D2_{cmp.focus_label}_km2": float(cmp.d2_focus[idx]),
                "D2_all_km2": float(cmp.d2_all[idx]),
                "n_SS": int(cmp.n_base[idx]),
                f"n_{cmp.focus_label}": int(cmp.n_focus[idx]),
                "boot95_SS_lo": float(cmp.boot_lo[idx]),
                "boot95_SS_hi": float(cmp.boot_hi[idx]),
                "drifter_pvalue": float(cmp.drifter_p[idx]),
            }
        )
    return pd.DataFrame(rows)


def _fit_rows(fits: dict[str, RegimeFit]) -> list[RegimeFit]:
    """Fits ordered pooled-first, then clusters alphabetically."""
    order = ["pooled"] + sorted(k for k in fits if k != "pooled")
    return [fits[k] for k in order if k in fits]


def lambda_fit_table(fits: dict[str, RegimeFit]) -> pd.DataFrame:
    """Exponential rate lambda per scope, fit from the slope of ln(D^2) vs t."""
    rows = []
    for f in _fit_rows(fits):
        rows.append({
            "scope": f.scope,
            "fit_lo_d": f.lambda_fit_lo_d,
            "fit_hi_d": f.lambda_fit_hi_d,
            "lambda_per_day": f.lambda_per_day,
            "efold_days": f.efold_days,
            "n": f.n_lambda,
        })
    return pd.DataFrame(rows)


def make_all(
    pairs: pd.DataFrame,
    comparisons: dict[str, InclusionComparison],
    fits: dict[str, RegimeFit] | None = None,
):
    _ensure_dir()
    counts = pair_counts(pairs)
    counts.to_latex(
        DISPERSION_STATS_DIR / "pair_counts.tex",
        index=False, float_format="%.1f",
    )
    if fits:
        lambda_fit_table(fits).to_latex(
            DISPERSION_STATS_DIR / "lambda_fit.tex",
            index=False, float_format="%.3g",
        )
    for name, cmp in comparisons.items():
        summary = inclusion_summary(cmp)
        if summary.empty:
            continue
        drift = "exact" if cmp.drifter_exact else f"{cmp.drifter_n_perm} samples"
        with (DISPERSION_STATS_DIR / f"{name}_inclusion.tex").open("w") as f:
            f.write(f"% {cmp.focus_label} inclusion at reference lags. "
                    f"drifter_pvalue = drifter-level permutation ({drift}).\n")
            f.write(summary.to_latex(index=False, float_format="%.3g"))
