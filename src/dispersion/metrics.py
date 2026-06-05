"""Relative-dispersion diagnostic on a pair-separation table.

Input is produced by ``dispersion.pairs.build_pairs``: rows are
``(pair_id, type_pair, cluster, t0, time, lag_h, r_km)``.

``relative_dispersion`` computes D^2(tau) = <r^2>_pairs at lag tau, with a
pair-bootstrap 95% confidence interval on the mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DispersionCurve:
    lag_h: np.ndarray
    d2_km2: np.ndarray
    n_pairs: np.ndarray
    sample_taus_h: np.ndarray = field(default_factory=lambda: np.array([]))
    sample_d2: np.ndarray = field(default_factory=lambda: np.array([]))
    boot_lo: np.ndarray = field(default_factory=lambda: np.array([]))
    boot_hi: np.ndarray = field(default_factory=lambda: np.array([]))


def _log_lag_grid(n: int = 100, min_lag_d: float = 0.1, max_lag_d: float = 100.0) -> np.ndarray:
    min_lag_h = min_lag_d * 24.0
    max_lag_h = max_lag_d * 24.0
    return np.unique(
        np.round(np.geomspace(min_lag_h, max_lag_h, n)).astype(int)
    ).astype(float)


def _pair_resampled(pair: pd.DataFrame, taus_h: np.ndarray) -> np.ndarray:
    """Linearly interpolate r_km onto ``taus_h``; NaN outside the pair span."""
    lag = pair["lag_h"].to_numpy()
    r = pair["r_km"].to_numpy()
    out = np.full_like(taus_h, np.nan, dtype=float)
    if lag.size < 2:
        return out
    inside = (taus_h >= lag[0]) & (taus_h <= lag[-1])
    if inside.any():
        out[inside] = np.interp(taus_h[inside], lag, r)
    return out


def relative_dispersion(
    pairs: pd.DataFrame,
    n_lag: int = 100,
    min_pairs: int = 3,
    n_boot: int = 5000,
    ci: float = 0.95,
    rng_seed: int = 0,
) -> DispersionCurve:
    """D^2(tau) averaged over a log-spaced lag grid.

    A pair-bootstrap 95 % central interval on D^2 is also returned
    (resampling the pair index with replacement ``n_boot`` times).
    """
    if pairs.empty:
        return DispersionCurve(np.array([]), np.array([]), np.array([]))
    taus_h = _log_lag_grid(n_lag)
    pair_groups = list(pairs.groupby("pair_id", sort=False))
    r2 = np.full((len(pair_groups), taus_h.size), np.nan, dtype=float)
    for i, (_, grp) in enumerate(pair_groups):
        r = _pair_resampled(grp, taus_h)
        r2[i] = r * r
    n_pairs = np.sum(np.isfinite(r2), axis=0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        d2 = np.nanmean(r2, axis=0)
        # Pair bootstrap on the (n_pairs, n_lag) matrix.
        n = r2.shape[0]
        boot_lo_full = np.full(taus_h.size, np.nan)
        boot_hi_full = np.full(taus_h.size, np.nan)
        if n >= 2 and n_boot > 0:
            rng = np.random.default_rng(rng_seed)
            boot = np.empty((n_boot, taus_h.size))
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                boot[b] = np.nanmean(r2[idx], axis=0)
            lo_q = 100 * (1 - ci) / 2.0
            hi_q = 100 * (1 + ci) / 2.0
            boot_lo_full = np.nanpercentile(boot, lo_q, axis=0)
            boot_hi_full = np.nanpercentile(boot, hi_q, axis=0)
    keep = n_pairs >= min_pairs
    return DispersionCurve(
        lag_h=taus_h[keep],
        d2_km2=d2[keep],
        n_pairs=n_pairs[keep],
        sample_taus_h=taus_h,
        sample_d2=r2,  # (n_pairs, n_lag)
        boot_lo=boot_lo_full[keep],
        boot_hi=boot_hi_full[keep],
    )
