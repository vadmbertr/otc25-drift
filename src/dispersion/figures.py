"""Figures for the dispersion analysis (inspired by Meyerjurgens 2020)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from utils.io_paths import PLOTS_DISPERSION_DIR

from .metrics import DispersionCurve, relative_dispersion
from .geom_test import InclusionComparison
from .regimes import RegimeFit, segments_for


LBL_BALLISTIC = r"Ballistic ($t^2$)"
LBL_EXP_T = r"Exponential ($e^{2\lambda t}$)"
LBL_RICH_T = r"Richardson ($t^3$)"
LBL_DIFF_T = r"Diffusive ($t^1$)"

_REGIME_COLOR = "#e41a1c"
_REGIME_COLORS = {
    "ballistic": _REGIME_COLOR,
    "exponential": _REGIME_COLOR,
    "richardson": _REGIME_COLOR,
    "diffusive": _REGIME_COLOR,
}
_REGIME_LS = {
    "ballistic": (0, (3, 1, 1, 1)),
    "exponential": "--",
    "richardson": "-.",
    "diffusive": ":",
}
_POOLED_ALPHA = 0.5
_T_XLIM_DAYS = (1e-1, 1e2)


def _ensure_dir():
    PLOTS_DISPERSION_DIR.mkdir(parents=True, exist_ok=True)


def _seg_x(x: np.ndarray, start: float, end: float | None) -> np.ndarray:
    hi = float(x.max()) if end is None else end
    return x[(x >= start) & (x <= hi)]


def _anchor(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """Empirical y at x0 (log-interp); falls back to first finite point."""
    m = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if m.sum() < 1:
        return float("nan")
    return float(np.interp(x0, x[m], y[m]))


def _add_powerlaw_seg(ax, x_data, y_data, slope, segment, label,
                      ls="--", color="0.4"):
    """Power-law guide y ~ (x/start)**slope over ``segment``, anchored to data.

    The guide is clipped to ``segment = (start, end)`` and passes through the
    empirical (x_data, y_data) at ``start`` so it lines up with the curve.
    """
    if segment is None:
        return
    start, end = segment
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    xs = _seg_x(x, start, end)
    if xs.size < 2 or start <= 0:
        return
    y0 = _anchor(x, y, start)
    if not np.isfinite(y0) or y0 <= 0:
        return
    ax.plot(xs, y0 * (xs / start) ** slope, ls=ls, color=color, lw=1.0,
            label=label)


def _add_exponential_seg(ax, x_data, y_data, lam_per_day, segment, label,
                         ls="--", color="0.4"):
    """y ~ y0 * exp(2 lambda (t - start)) over the time ``segment`` (days)."""
    if segment is None or not np.isfinite(lam_per_day):
        return
    start, end = segment
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    xs = _seg_x(x, start, end)
    if xs.size < 2:
        return
    y0 = _anchor(x, y, start)
    if not np.isfinite(y0) or y0 <= 0:
        return
    ax.plot(xs, y0 * np.exp(2.0 * lam_per_day * (xs - start)), ls=ls,
            color=color, lw=1.0, label=label)


def _add_d2t_regimes(ax, lag_d, d2, fit: RegimeFit | None, segments: dict | None):
    """Draw the ballistic / exponential / Richardson / diffusive D^2(t) guides.

    ``segments`` maps regime name -> (start_day, end_day); a regime is drawn
    only if present.  Only the exponential curve uses the fitted ``lambda``.
    """
    d = segments or {}
    _add_powerlaw_seg(ax, lag_d, d2, 2.0, d.get("ballistic"), LBL_BALLISTIC,
                      ls=_REGIME_LS["ballistic"], color=_REGIME_COLORS["ballistic"])
    if fit is not None:
        _add_exponential_seg(ax, lag_d, d2, fit.lambda_per_day, d.get("exponential"),
                             LBL_EXP_T, ls=_REGIME_LS["exponential"],
                             color=_REGIME_COLORS["exponential"])
    _add_powerlaw_seg(ax, lag_d, d2, 3.0, d.get("richardson"), LBL_RICH_T,
                      ls=_REGIME_LS["richardson"], color=_REGIME_COLORS["richardson"])
    _add_powerlaw_seg(ax, lag_d, d2, 1.0, d.get("diffusive"), LBL_DIFF_T,
                      ls=_REGIME_LS["diffusive"], color=_REGIME_COLORS["diffusive"])


def plot_dispersion(
    curves: dict[str, DispersionCurve], out_name: str, title: str,
    fit: RegimeFit | None = None, segments=None,
):
    _ensure_dir()
    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    keys = [k for k in curves if curves[k].lag_h.size]
    for k in keys:
        c = curves[k]
        if c.boot_lo.size and c.boot_hi.size:
            t_d = c.lag_h / 24.0
            ok = np.isfinite(c.boot_lo) & np.isfinite(c.boot_hi) \
                 & (c.boot_lo > 0) & (c.boot_hi > 0)
            if ok.any():
                ax.fill_between(
                    t_d[ok], c.boot_lo[ok], c.boot_hi[ok],
                    color="0.7", alpha=0.4,
                    label=f"{k} 95% CI",
                )
        ax.loglog(c.lag_h / 24.0, c.d2_km2, label=k, color="k", lw=1.6,
                  alpha=_POOLED_ALPHA)
        ax_n.semilogx(c.lag_h / 24.0, c.n_pairs, color="k", lw=1.0,
                      alpha=_POOLED_ALPHA)
    if keys:
        anchor = curves[keys[0]]
        if anchor.lag_h.size:
            _add_d2t_regimes(ax, anchor.lag_h / 24.0, anchor.d2_km2, fit, segments)
    ax.set_xlim(*_T_XLIM_DAYS)
    ax.set_ylabel("$D^2$ (km$^2$)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax_n.set_xlabel("lag (days)")
    ax_n.set_ylabel("# pairs")
    ax_n.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DISPERSION_DIR / out_name, dpi=300)
    plt.close(fig)


def _cluster_colors(cluster_ids):
    cmap = plt.get_cmap("viridis")
    n = len(cluster_ids)
    return {
        c: cmap(0.15 + 0.7 * i / max(n - 1, 1))
        for i, c in enumerate(cluster_ids)
    }


def _facet_grid(n: int, panel_size: tuple[float, float] = (4.5, 3.6)):
    """Create a sqrt-ish grid of subplots with shared x and y axes.

    Returns ``(fig, axes_flat)`` with extra axes hidden.
    """
    cols = max(1, int(np.ceil(np.sqrt(n))))
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(panel_size[0] * cols, panel_size[1] * rows),
        sharex=True, sharey=True,
        squeeze=False,
    )
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def plot_dispersion_facets(
    per_cluster: dict[str, DispersionCurve],
    pooled: DispersionCurve,
    out_name: str,
    title: str,
    fits: dict[str, RegimeFit] | None = None,
):
    """Faceted D^2(t): one panel per deployment cluster.

    Each panel shows individual-pair r^2 samples as scatter, the
    cluster-averaged D^2 curve, the pooled curve (faint reference),
    and t^2 / t^3 power-law guides.
    """
    _ensure_dir()
    keys = [k for k in sorted(per_cluster) if per_cluster[k].lag_h.size]
    if not keys:
        return
    fig, axes = _facet_grid(len(keys))
    colors = _cluster_colors(keys)
    pooled_t_d = pooled.lag_h / 24.0 if pooled.lag_h.size else np.array([])
    for ax, k in zip(axes, keys):
        c = per_cluster[k]
        col = colors[k]
        # Individual-pair scatter: sample_d2 is (n_pairs, n_lag) of r^2 in km^2.
        if c.sample_d2.size and c.sample_taus_h.size:
            taus_d = c.sample_taus_h / 24.0
            t_grid = np.broadcast_to(taus_d, c.sample_d2.shape)
            mask = np.isfinite(c.sample_d2) & (c.sample_d2 > 0)
            if mask.any():
                ax.scatter(
                    t_grid[mask], c.sample_d2[mask],
                    s=6, color=col, alpha=0.25, linewidths=0,
                )
        # Per-cluster pair-bootstrap 95% CI band.
        if c.boot_lo.size and c.boot_hi.size:
            t_d_cl = c.lag_h / 24.0
            ok = np.isfinite(c.boot_lo) & np.isfinite(c.boot_hi) \
                 & (c.boot_lo > 0) & (c.boot_hi > 0)
            if ok.any():
                ax.fill_between(
                    t_d_cl[ok], c.boot_lo[ok], c.boot_hi[ok],
                    color=col, alpha=0.25,
                    label="95% CI",
                )
        # Cluster-average curve.
        ax.loglog(c.lag_h / 24.0, c.d2_km2, color=col, lw=1.6, label=f"{k} mean")
        # Pooled overlay (black solid so it never clashes with the guides).
        if pooled.lag_h.size:
            ax.loglog(pooled_t_d, pooled.d2_km2, color="k", lw=1.0,
                      alpha=_POOLED_ALPHA, label="pooled")
        # Regime guides over this cluster's user-defined lag segments.
        if c.lag_h.size:
            fit = fits.get(k) if fits else None
            _add_d2t_regimes(ax, c.lag_h / 24.0, c.d2_km2, fit, segments_for(k))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*_T_XLIM_DAYS)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(k, fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
    # Outer labels: leftmost column gets y, bottom row gets x.
    for ax in axes:
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("$D^2$ (km$^2$)")
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("lag (days)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(PLOTS_DISPERSION_DIR / out_name, dpi=300)
    plt.close(fig)


def plot_inclusion_comparison(cmp: InclusionComparison, out_name: str, focus_text: str):
    """Relative dispersion of one platform's SPOT-pairs vs the SS baseline.

    ``focus_text`` is a short human label for the focus stratum, e.g.
    ``"SPOT-MARGET"``.  The top panel shows the relative D^2 curves each with a
    bootstrap 95% CI envelope; the bottom panel shows the drifter-level
    permutation p-value vs lag, with the minimum achievable p (1/N_permutations).
    """
    _ensure_dir()
    if cmp is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax, ax_p = axes
    t_d = cmp.lag_h / 24.0
    has_base = cmp.n_base > 0
    has_focus = cmp.n_focus > 0

    def _band(mask, lo, hi, color):
        m = mask & np.isfinite(lo) & np.isfinite(hi)
        if m.any():
            ax.fill_between(t_d[m], lo[m], hi[m], color=color, alpha=0.18)

    _band(has_base, cmp.boot_lo, cmp.boot_hi, "tab:blue")
    _band(has_focus, cmp.focus_lo, cmp.focus_hi, "tab:orange")
    ax.loglog(t_d[has_base], cmp.d2_base[has_base], color="tab:blue", lw=1.6,
              label="SPOT-SPOT")
    ax.loglog(t_d[has_focus], cmp.d2_focus[has_focus], color="tab:orange", lw=1.6,
              label=focus_text)
    ax.set_ylabel("$D^2$ (km$^2$)")
    ax.set_title(f"{focus_text} inclusion: relative dispersion")
    ax.grid(True, which="both", alpha=0.3)
    # One generic "95% CI" legend entry rather than one per stratum.
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.6", alpha=0.3))
    labels.append("95% CI")
    ax.legend(handles, labels, fontsize=8)

    # Bottom panel: drifter-level permutation p vs lag (log y).
    p_min = 1.0 / cmp.drifter_n_perm if cmp.drifter_n_perm > 0 else None
    y_lo = max(1e-4, min(0.04, 0.5 * p_min)) if p_min else 1e-3
    d_ok = np.isfinite(cmp.drifter_p)
    if d_ok.any():
        ax_p.plot(t_d[d_ok], np.clip(cmp.drifter_p[d_ok], y_lo, 1.0),
                  "-", color="tab:orange", lw=1.2)
    ax_p.set_xscale("log")
    ax_p.set_yscale("log")
    ax_p.set_ylim(y_lo, 1.0)
    ax_p.axhline(0.05, color="0.5", lw=0.8, ls=":")
    ax_p.text(t_d[-5], 0.05, " p=0.05", fontsize=7, color="0.4", ha="left", va="bottom")
    if p_min is not None:  # minimum achievable p = 1 / N permutations
        ax_p.axhline(p_min, color="tab:red", lw=0.8, ls="--")
        ax_p.text(t_d[-5], p_min, f" min $p$=1/{cmp.drifter_n_perm}",
                  fontsize=7, color="tab:red", ha="left", va="bottom")
    ax_p.set_xlabel("lag (days)")
    ax_p.set_ylabel("$p$-value")
    ax_p.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DISPERSION_DIR / out_name, dpi=300)
    plt.close(fig)


def _order_spots_by_spread(spots: dict[str, pd.DataFrame]) -> list[str]:
    """Order SPOT ids along the principal axis of their mean positions.

    With the shading running along the line the drifters spread out on,
    neighbouring colours correspond to neighbouring tracks.  Longitude is scaled
    by cos(lat) so the axis is in physical (metric) proportions; the axis is
    oriented northward for a consistent light-to-dark gradient.
    """
    ids = list(spots)
    means = {s: (spots[s]["lon_smooth"].mean(), spots[s]["lat_smooth"].mean()) for s in ids}
    if len(ids) < 3:
        return sorted(ids, key=lambda s: (means[s][1], means[s][0]))
    lat0 = np.deg2rad(np.mean([means[s][1] for s in ids]))
    pts = np.array([[means[s][0] * np.cos(lat0), means[s][1]] for s in ids])
    centered = pts - pts.mean(axis=0)
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    if axis[1] < 0:  # orient toward north
        axis = -axis
    proj = centered @ axis
    return [ids[i] for i in np.argsort(proj)]


# Per-type base colormaps for the trajectory diagnostic (SPOT is the baseline).
_TRAJ_TYPE_CMAPS = {
    "SPOT": "Blues",
    "MARGET": "Oranges",
    "MELODI": "Greens",
    "OMB": "Purples",
}


def _traj_label(drifter_id: str, dtype: str) -> str:
    """Compact, type-tagged legend label (e.g. 'SPOT 4496232', 'MELODI-15')."""
    if drifter_id.startswith("OTC25-"):
        return drifter_id[len("OTC25-"):]
    return f"{dtype} {drifter_id[-7:]}"


def plot_cluster_trajectories(
    drifters: pd.DataFrame,
    cluster_id: str,
    members_by_type: dict[str, list[str]],
    t0: pd.Timestamp,
    lag_max_h: float,
    out_name: str,
):
    """Map + lat/lon-vs-lag panels of a deployment's tracks over the first lags.

    Used to inspect the platform-inclusion signal: the map shows the
    relative-position structure (parallel tracks with a sustained offset vs. a
    transient excursion), and the lat/lon panels show whether the platforms
    oscillate in phase.  Each platform type is drawn in its own colormap, shaded
    within type along the spatial spread; SPOT is the baseline.  Drifters with no
    data in the ``[0, lag_max_h]`` window are dropped.
    """
    _ensure_dir()

    def get(d):
        sub = drifters[drifters["drifter_id"] == d][
            ["time", "lat_smooth", "lon_smooth"]
        ].dropna().copy()
        sub["lag_h"] = (sub["time"] - t0).dt.total_seconds() / 3600.0
        return sub[(sub["lag_h"] >= 0) & (sub["lag_h"] <= lag_max_h)] \
            .sort_values("lag_h").reset_index(drop=True)

    # Window + drop-empty per type, preserving SPOT first.
    tracks_by_type: dict[str, dict[str, pd.DataFrame]] = {}
    for dtype in sorted(members_by_type, key=lambda t: (t != "SPOT", t)):
        tracks = {d: get(d) for d in members_by_type[dtype]}
        tracks = {d: v for d, v in tracks.items() if not v.empty}
        if tracks:
            tracks_by_type[dtype] = tracks
    if not tracks_by_type:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_map, ax_lat, ax_lon = axes

    # Non-SPOT tracks are the reference platforms; annotate time only when there
    # is exactly one of them in the cluster (keeps the map readable).
    non_spot_ids = [d for t, tr in tracks_by_type.items() if t != "SPOT" for d in tr]

    count_summary = []
    for dtype, tracks in tracks_by_type.items():
        ids = _order_spots_by_spread(tracks)  # shading follows spatial order
        cmap = plt.get_cmap(_TRAJ_TYPE_CMAPS.get(dtype, "Greys"))
        shades = ([cmap(0.7)] if len(ids) == 1
                  else cmap(np.linspace(0.4, 0.9, len(ids))))
        lw = 1.0 if dtype == "SPOT" else 1.8
        for color, d in zip(shades, ids):
            sub = tracks[d]
            label = _traj_label(d, dtype)
            ax_map.plot(sub["lon_smooth"], sub["lat_smooth"], color=color, lw=lw,
                        alpha=0.85, label=label)
            ax_lat.plot(sub["lag_h"], sub["lat_smooth"], color=color, lw=lw,
                        alpha=0.85, label=label)
            ax_lon.plot(sub["lag_h"], sub["lon_smooth"], color=color, lw=lw,
                        alpha=0.85, label=label)
            if dtype != "SPOT" and len(non_spot_ids) == 1:
                for lag_marker in range(0, int(lag_max_h) + 1, 6):
                    row = sub.iloc[(sub["lag_h"] - lag_marker).abs().argsort()[:1]]
                    if len(row):
                        ax_map.annotate(
                            f"{lag_marker}h",
                            (row["lon_smooth"].iloc[0], row["lat_smooth"].iloc[0]),
                            xytext=(5, 3), textcoords="offset points",
                            fontsize=7, color=cmap(0.95),
                        )
        count_summary.append(f"{len(ids)} {dtype}")

    ax_map.set_xlabel("Longitude (deg E)")
    ax_map.set_ylabel("Latitude (deg N)")
    ax_map.set_title(f"{cluster_id} map, deployment + {int(lag_max_h)} h")
    ax_map.grid(True, alpha=0.3)
    ax_map.legend(fontsize=7, loc="best")
    ax_map.set_aspect("equal", adjustable="datalim")

    for ax, ylabel, title in (
        (ax_lat, "Latitude (deg N)", "Latitude vs lag"),
        (ax_lon, "Longitude (deg E)", "Longitude vs lag"),
    ):
        ax.set_xlabel("lag (h)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    sup = (f"{cluster_id} deployment: {', '.join(count_summary)} "
           f"(lag 0 - {int(lag_max_h)} h)")
    fig.suptitle(sup, fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS_DISPERSION_DIR / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_all(
    pairs: pd.DataFrame,
    comparisons: dict[str, InclusionComparison],
    drifters: pd.DataFrame | None = None,
    fits: dict[str, RegimeFit] | None = None,
):
    """Generate every dispersion figure from the cluster-pair table.

    The relative-dispersion D^2(t) diagnostic uses SPOT-SPOT pairs only.
    Whether including MARGETs would change the result is the subject of
    the dedicated MARGET inclusion plot driven by ``comparison``.  If
    ``drifters`` (the smoothed hourly table) is provided, a per-cluster
    MARGET-vs-SPOT trajectory diagnostic is also produced for every cluster
    that contains both platform types -- useful to read the inclusion-test
    signal in physical space.
    """
    if pairs.empty:
        return
    ss = pairs[pairs["type_pair"] == "SS"]
    if ss.empty:
        return

    # Per-cluster D^2 curves (SS only).
    cluster_curves: dict[str, DispersionCurve] = {}
    for cluster_id, sub in ss.groupby("cluster", sort=True):
        cluster_curves[cluster_id] = relative_dispersion(sub)

    # Pooled curve (SS only).
    pooled = relative_dispersion(ss)
    pooled_fit = fits.get("pooled") if fits else None
    plot_dispersion(
        {"all clusters": pooled},
        "dispersion_pooled.png",
        "Relative dispersion (SPOT-SPOT, all clusters)",
        fit=pooled_fit, segments=segments_for("pooled"),
    )

    # Faceted per-deployment views with individual-pair scatter.
    plot_dispersion_facets(
        cluster_curves, pooled, "dispersion_per_cluster.png",
        "Relative dispersion per deployment cluster (SPOT-SPOT)",
        fits=fits,
    )

    # Platform-inclusion tests (separate diagnostics), one per focus type.
    for name, cmp in comparisons.items():
        focus_text = cmp.focus_label.replace("S-", "SPOT-")
        plot_inclusion_comparison(cmp, f"{name}_inclusion.png", focus_text)

    # Per-cluster trajectory diagnostic over the first 48 h of every deployment.
    # Reads pair metadata (cluster -> deploy time, member ids) from the pair
    # table so it never goes out of sync with build_cluster_pairs.  Each platform
    # type is drawn in its own colour; SPOT is the baseline.
    if drifters is None:
        return
    type_by_id = (
        drifters.drop_duplicates("drifter_id").set_index("drifter_id")["type"].to_dict()
    )
    for cluster_id, sub in pairs.groupby("cluster", sort=True):
        members = set(sub["drifter_a"]).union(sub["drifter_b"])
        members_by_type: dict[str, list[str]] = {}
        for d in sorted(members):
            members_by_type.setdefault(type_by_id.get(d, "?"), []).append(d)
        t0 = pd.Timestamp(sub["t0"].iloc[0])
        plot_cluster_trajectories(
            drifters,
            cluster_id=cluster_id,
            members_by_type=members_by_type,
            t0=t0,
            lag_max_h=48.0,
            out_name=f"{cluster_id.lower()}_trajectories_48h.png",
        )
