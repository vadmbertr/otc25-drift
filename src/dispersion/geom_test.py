"""Test whether including a non-SPOT platform changes the dispersion.

Generic two-stratum inclusion test (cluster pairs only, where t0 is
well-defined).  The baseline is SPOT-SPOT (``SS``); the focus stratum is the
SPOT-vs-other pairs of one platform type (``S-MARGET``, ``S-MELODI`` or
``S-OMB``).

1. Stratify pairs into ``baseline_label`` and ``focus_label`` and compute
   D^2(tau) for each stratum, plus a baseline pair-bootstrap 95% CI envelope
   (a descriptive band only -- no test is derived from it).

2. Drifter-level permutation test (the inference): pairs are not independent --
   the focus pairs in a cluster all share the focus drifter -- so the
   exchangeable unit under the null ("the focus drifter behaves like a SPOT") is
   the *drifter*.  Within each cluster we relabel which single drifter is "focus"
   (its pairs form the focus stratum, the rest the baseline) and recompute
   ``mean(r^2|focus) - mean(r^2|base)`` at reference lags.  The permutation
   universe is the product of per-cluster drifter counts; it is enumerated
   exactly when small.  With few focus drifters this test has little power
   (e.g. one OMB -> min p = 1/6), which is the point.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import _pair_resampled, _log_lag_grid


@dataclass
class InclusionComparison:
    focus_label: str
    baseline_label: str
    lag_h: np.ndarray
    d2_base: np.ndarray
    d2_focus: np.ndarray
    d2_all: np.ndarray
    n_base: np.ndarray
    n_focus: np.ndarray
    boot_lo: np.ndarray  # baseline bootstrap 2.5% (CI envelope, matches lag_h)
    boot_hi: np.ndarray  # baseline bootstrap 97.5%
    focus_lo: np.ndarray  # focus-stratum bootstrap CI envelope
    focus_hi: np.ndarray
    perm_lags_h: np.ndarray  # reference lags reported in the table
    drifter_p: np.ndarray  # drifter-level two-sided p at each lag (matches lag_h)
    drifter_n_perm: int  # size of the drifter permutation null used
    drifter_exact: bool  # True if the drifter universe was enumerated exactly


def _r2_matrix(pairs: pd.DataFrame, taus_h: np.ndarray) -> np.ndarray:
    rows = []
    for _, grp in pairs.groupby("pair_id", sort=False):
        rows.append(_pair_resampled(grp, taus_h) ** 2)
    if not rows:
        return np.empty((0, taus_h.size))
    return np.vstack(rows)


def _mean_finite(arr: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=0) if arr.size else np.full(arr.shape[1:], np.nan)


def _boot_ci(
    r2: np.ndarray, n_boot: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair-bootstrap 95% CI envelope on the stratum mean D^2 (descriptive)."""
    n_lag = r2.shape[1] if r2.ndim == 2 else 0
    n = r2.shape[0]
    if n < 2 or n_lag == 0:
        return np.full(n_lag, np.nan), np.full(n_lag, np.nan)
    boot = np.empty((n_boot, n_lag))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for b in range(n_boot):
            boot[b] = np.nanmean(r2[rng.integers(0, n, size=n)], axis=0)
        return (np.nanpercentile(boot, 2.5, axis=0),
                np.nanpercentile(boot, 97.5, axis=0))


def _drifter_perm_test(
    pairs: pd.DataFrame,
    focus_label: str,
    baseline_label: str,
    lags: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    max_enum: int = 200_000,
) -> tuple[np.ndarray, int, bool]:
    """Permute the focus *drifter* within each cluster (pseudo-replication aware).

    In each cluster that contains the focus type, exactly one drifter is the
    focus platform; the others are SPOTs.  Under the null it is exchangeable with
    those SPOTs, so a "designation" picks one drifter per cluster to be focus --
    its within-cluster pairs form the focus stratum, the remaining pairs the
    baseline -- and the statistic ``mean(r^2|focus) - mean(r^2|base)`` is
    recomputed at every lag.  The universe is the product of per-cluster drifter
    counts; enumerated exactly when ``<= max_enum``, else sampled ``n_perm``
    times.

    Returns ``(p, n_perm_used, exact)`` where ``p`` is the two-sided p-value at
    each lag (NaN where the statistic is undefined).
    """
    p = np.full(lags.size, np.nan)
    if lags.size == 0:
        return p, 0, True
    sub_all = pairs[pairs["type_pair"].isin([baseline_label, focus_label])]
    clusters = sorted(sub_all[sub_all["type_pair"] == focus_label]["cluster"].unique())

    cl_drifters: list[list[str]] = []
    cl_recs: list[list[tuple[str, str, np.ndarray]]] = []
    cl_focus: list[str] = []
    for cl in clusters:
        sub = sub_all[sub_all["cluster"] == cl]
        spot_ids: set[str] = set()
        recs: list[tuple[str, str, np.ndarray]] = []
        for _, grp in sub.groupby("pair_id", sort=False):
            a = grp["drifter_a"].iloc[0]
            b = grp["drifter_b"].iloc[0]
            recs.append((a, b, _pair_resampled(grp, lags) ** 2))
            if grp["type_pair"].iloc[0] == baseline_label:
                spot_ids.update((a, b))
        drifters = sorted({a for a, _, _ in recs} | {b for _, b, _ in recs})
        focus_ids = [d for d in drifters if d not in spot_ids]
        if len(focus_ids) != 1:
            continue  # need exactly one focus drifter to relabel cleanly
        cl_drifters.append(drifters)
        cl_recs.append(recs)
        cl_focus.append(focus_ids[0])
    if not cl_drifters:
        return p, 0, True

    def stat(choice: tuple[str, ...]) -> np.ndarray:
        f_rows, b_rows = [], []
        for recs, g in zip(cl_recs, choice):
            for a, b, r2 in recs:
                (f_rows if g in (a, b) else b_rows).append(r2)
        f = np.nanmean(np.vstack(f_rows), axis=0) if f_rows else np.full(lags.size, np.nan)
        base = np.nanmean(np.vstack(b_rows), axis=0) if b_rows else np.full(lags.size, np.nan)
        return f - base

    total = int(np.prod([len(d) for d in cl_drifters]))
    exact = total <= max_enum
    ge = np.zeros(lags.size)
    tol = 1e-12
    with warnings.catch_warnings():
        # all-NaN slices (a stratum empty at long lags) -> NaN, expected.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        obs = stat(tuple(cl_focus))
        finite = np.isfinite(obs)
        if exact:
            for choice in itertools.product(*cl_drifters):
                ge += np.abs(stat(choice)) >= np.abs(obs) - tol
            n_used = total
            p[finite] = ge[finite] / n_used
        else:
            for _ in range(n_perm):
                choice = tuple(d[rng.integers(len(d))] for d in cl_drifters)
                ge += np.abs(stat(choice)) >= np.abs(obs) - tol
            n_used = n_perm
            # (count + 1) / (n + 1): the observed designation is itself one draw.
            p[finite] = (ge[finite] + 1.0) / (n_used + 1.0)
    return p, n_used, exact


def compare(
    pairs: pd.DataFrame,
    focus_label: str,
    baseline_label: str = "SS",
    n_boot: int = 5000,
    n_lag: int = 60,
    perm_lags_h: tuple[float, ...] = (24.0, 24.0 * 7, 24.0 * 30),
    n_perm: int = 10000,
    rng_seed: int = 0,
) -> InclusionComparison | None:
    """D^2 with a baseline bootstrap CI envelope, plus the
    drifter-level permutation test.  Returns ``None`` if a stratum is empty."""
    rng = np.random.default_rng(rng_seed)

    base = pairs[pairs["type_pair"] == baseline_label]
    focus = pairs[pairs["type_pair"] == focus_label]
    if base.empty or focus.empty:
        return None

    max_lag = float(min(base["lag_h"].max(), focus["lag_h"].max()))
    taus_h = _log_lag_grid(n_lag, max_lag_d=max_lag / 24.0)

    r2_base = _r2_matrix(base, taus_h)
    r2_focus = _r2_matrix(focus, taus_h)
    r2_all = np.vstack([r2_base, r2_focus])

    d2_base = _mean_finite(r2_base)
    d2_focus = _mean_finite(r2_focus)
    d2_all = _mean_finite(r2_all)
    n_base = np.sum(np.isfinite(r2_base), axis=0)
    n_focus = np.sum(np.isfinite(r2_focus), axis=0)

    # Pair-bootstrap 95% CI envelopes on the SS and focus stratum means.
    boot_lo, boot_hi = _boot_ci(r2_base, n_boot, rng)
    focus_lo, focus_hi = _boot_ci(r2_focus, n_boot, rng)

    # Drifter-level permutation (pseudo-replication aware) at every lag.
    perm_lags = np.array([t for t in perm_lags_h if t <= max_lag], dtype=float)
    drifter_p, drifter_n_perm, drifter_exact = _drifter_perm_test(
        pairs, focus_label, baseline_label, taus_h, n_perm, rng,
    )

    return InclusionComparison(
        focus_label=focus_label,
        baseline_label=baseline_label,
        lag_h=taus_h,
        d2_base=d2_base,
        d2_focus=d2_focus,
        d2_all=d2_all,
        n_base=n_base,
        n_focus=n_focus,
        boot_lo=boot_lo,
        boot_hi=boot_hi,
        focus_lo=focus_lo,
        focus_hi=focus_hi,
        perm_lags_h=perm_lags,
        drifter_p=drifter_p,
        drifter_n_perm=drifter_n_perm,
        drifter_exact=drifter_exact,
    )
