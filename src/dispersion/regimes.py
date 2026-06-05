"""Exponential dispersion-rate fit and the D^2(t) regime-guide segments.

The relative dispersion D^2(t) is annotated with reference curves for the four
classic regimes:

    Diffusive    D^2 ~ t^1
    Ballistic    D^2 ~ t^2
    Richardson   D^2 ~ t^3
    Exponential  D^2 ~ e^{2 lambda t}

Each regime is drawn only over a user-chosen lag range, set per scope ("pooled"
and each cluster) in ``REGIME_SEGMENTS``; omit a regime to hide it.  The only
*fitted* quantity is the exponential rate ``lambda``, obtained from the slope of
``ln(D^2)`` vs time over the ``"exponential"`` lag range of that scope
(``fit_lambda_d2t``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import relative_dispersion


# --- hand-tune knobs ------------------------------------------------------
# Per-scope reference-curve lag ranges, in DAYS: ``regime -> (start, end)``;
# ``end=None`` extends to the last lag.  A regime is drawn only if listed, so
# omit one to hide it.  The ``"exponential"`` range is ALSO the lambda fit
# window.  Re-run after edits.
REGIME_SEGMENTS: dict[str, dict[str, tuple[float, float | None]]] = {
    "pooled": {"exponential": (0.1, 6), "ballistic": (6, None)},
    "C01": {"exponential": (0.25, 5), "ballistic": (5, None)},
    "C02": {"richardson": (1.5, None)},
    "C03": {"exponential": (0, 4), "richardson": (4, None)},
    "C04": {"exponential": (0, 10), "richardson": (10, None)},
}


@dataclass
class RegimeFit:
    scope: str
    lambda_per_day: float
    efold_days: float
    n_lambda: int
    lambda_fit_lo_d: float
    lambda_fit_hi_d: float


def fit_lambda_d2t(
    curve, t_lo_d: float, t_hi_d: float | None,
) -> tuple[float, int]:
    """Exponential rate from the slope of ``ln(D^2)`` vs time.

    In the exponential regime ``D^2(t) ~ e^{2 lambda t}``, so a straight-line
    fit of ``ln(D^2)`` against lag (days) over ``[t_lo, t_hi]`` has slope
    ``2*lambda``.  Points are weighted by the number of pairs.  Returns
    ``(lambda_per_day, n_points)``; NaN if fewer than 3 points fall in the
    window.  ``t_hi=None`` extends to the last lag.
    """
    if curve.lag_h.size == 0:
        return float("nan"), 0
    t = curve.lag_h / 24.0
    d2 = curve.d2_km2
    hi = float(t.max()) if t_hi_d is None else t_hi_d
    sel = (t >= t_lo_d) & (t <= hi) & np.isfinite(d2) & (d2 > 0)
    if sel.sum() < 3:
        return float("nan"), 0
    w = np.sqrt(curve.n_pairs[sel].astype(float)) if curve.n_pairs.size else None
    slope = float(np.polyfit(t[sel], np.log(d2[sel]), 1, w=w)[0])
    return slope / 2.0, int(sel.sum())


def segments_for(scope: str) -> dict[str, tuple[float, float | None]]:
    """Per-scope regime lag ranges (days); empty dict if the scope is unset."""
    return REGIME_SEGMENTS.get(scope, {})


def _fit_scope(scope: str, sub: pd.DataFrame) -> RegimeFit:
    curve = relative_dispersion(sub)
    seg = segments_for(scope).get("exponential")
    if seg is None:
        return RegimeFit(scope, float("nan"), float("nan"), 0,
                         float("nan"), float("nan"))
    lo, hi = seg
    lam_day, n_lam = fit_lambda_d2t(curve, lo, hi)
    efold = 1.0 / lam_day if lam_day and np.isfinite(lam_day) else float("nan")
    return RegimeFit(
        scope=scope,
        lambda_per_day=lam_day,
        efold_days=efold,
        n_lambda=n_lam,
        lambda_fit_lo_d=float(lo),
        lambda_fit_hi_d=float(hi) if hi is not None else float("nan"),
    )


def compute_all(ss_pairs: pd.DataFrame) -> dict[str, RegimeFit]:
    """Fit ``pooled`` + one entry per cluster from the SPOT-SPOT pair table."""
    fits: dict[str, RegimeFit] = {}
    if ss_pairs.empty:
        return fits
    fits["pooled"] = _fit_scope("pooled", ss_pairs)
    for cluster_id, sub in ss_pairs.groupby("cluster", sort=True):
        fits[cluster_id] = _fit_scope(cluster_id, sub)
    return fits
