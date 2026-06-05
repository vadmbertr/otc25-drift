"""Figures for the preprocessing step."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

from utils.io_paths import DEPLOY_TIME_CSV, PLOTS_DIR


def _ensure_dir():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _type_cmaps():
    return {
        "SPOT": plt.get_cmap("Blues"),
        "MARGET": plt.get_cmap("Oranges"),
        "MELODI": plt.get_cmap("Greens"),
        "OMB": plt.get_cmap("Purples"),
    }


TYPE_CMAPS = _type_cmaps()


def counts_per_day(raw: pd.DataFrame, filt: pd.DataFrame, drifter_type: str):
    _ensure_dir()
    raw_t = raw[raw["type"] == drifter_type]
    filt_t = filt[filt["type"] == drifter_type]
    raw_per_day = raw_t.groupby(raw_t["time"].dt.floor("1D")).size()
    filt_per_day = filt_t.groupby(filt_t["time"].dt.floor("1D")).size()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(raw_per_day.index, raw_per_day.values, label="raw", lw=1.0)
    ax.plot(filt_per_day.index, filt_per_day.values, label="filtered (hourly grid)", lw=1.0)
    ax.set_title(f"{drifter_type}: observation counts per day")
    ax.set_xlabel("date")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"counts_{drifter_type.lower()}.png", dpi=300)
    plt.close(fig)


def sampling_histogram(raw: pd.DataFrame, drifter_type: str):
    _ensure_dir()
    sub = raw[raw["type"] == drifter_type].sort_values(["drifter_id", "time"])
    dts = []
    for _, g in sub.groupby("drifter_id"):
        d = g["time"].diff().dt.total_seconds().dropna() / 60.0
        dts.append(d.to_numpy())
    if not dts:
        return
    dts = np.concatenate(dts)
    dts = dts[dts > 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.clip(dts, 0, 240), bins=60)
    ax.set_title(f"{drifter_type}: raw inter-fix interval (clipped at 4 h)")
    ax.set_xlabel("interval (minutes)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"sampling_{drifter_type.lower()}.png", dpi=300)
    plt.close(fig)


def trajectories(raw: pd.DataFrame, smooth: pd.DataFrame):
    _ensure_dir()
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.lines as mlines

    fig = plt.figure(figsize=(11, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Raw fixes in light gray, dots only.
    ax.scatter(
        raw["lon"], raw["lat"], s=0.06, color="k", marker=",", alpha=.1,
        transform=ccrs.PlateCarree(),
    )

    type_cmaps = TYPE_CMAPS
    for dtype, type_sub in smooth.groupby("type"):
        drifter_ids = list(type_sub["drifter_id"].unique())
        n = len(drifter_ids)
        cmap = type_cmaps.get(dtype, plt.get_cmap("Greys"))
        for i, did in enumerate(drifter_ids):
            color = cmap(0.4 + 0.5 * (i / max(n - 1, 1)))
            sub = type_sub[type_sub["drifter_id"] == did]
            for _, seg in sub.groupby("segment_id"):
                ax.plot(
                    seg["lon_smooth"], seg["lat_smooth"],
                    color=color, lw=0.8, transform=ccrs.PlateCarree(),
                )

    handles = [mlines.Line2D(
        [], [], color="k", marker=".", linestyle="None", markersize=3, label="raw",
    )]
    for t in sorted(smooth["type"].unique()):
        cmap = type_cmaps.get(t, plt.get_cmap("Greys"))
        handles.append(mlines.Line2D([], [], color=cmap(0.7), label=t))

    ax.set_title("Drifter trajectories — raw fixes (gray) and smoothed segments")
    ax.set_extent([-25, 21, 20, 74], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, lw=0.5)
    ax.gridlines(draw_labels=True, alpha=0.3)
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "trajectories.png", dpi=300)
    plt.close(fig)


def _deploy_clusters() -> pd.DataFrame:
    """deploy_time.csv grouped on (deploy_time, lat, lon) -> id, type, cluster, deploy_time."""
    df = pd.read_csv(DEPLOY_TIME_CSV, sep=r"\s+")
    df["deploy_time"] = pd.to_datetime(df["deploy_time"], utc=True, format="mixed")
    df["id"] = df["id"].astype(str)
    df["cluster"] = (
        df.groupby(["deploy_time", "lat", "lon"], sort=False).ngroup().add(1)
        .map(lambda i: f"C{i:02d}")
    )
    return df


def trajectories_per_cluster(raw: pd.DataFrame, smooth: pd.DataFrame):
    """One panel per deployment cluster: raw fixes (gray) and smoothed tracks,
    over the first day after the cluster deploy time, colored by platform type."""
    _ensure_dir()
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.lines as mlines

    clusters = _deploy_clusters()
    id2cluster = dict(zip(clusters["id"], clusters["cluster"]))
    deploy_by_cluster = clusters.groupby("cluster")["deploy_time"].first()
    cluster_ids = sorted(deploy_by_cluster.index)

    raw = raw.assign(cluster=raw["drifter_id"].map(id2cluster))
    smooth = smooth.assign(cluster=smooth["drifter_id"].map(id2cluster))

    n = len(cluster_ids)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(7 * ncol, 5 * nrow),
        subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False,
    )
    axes = axes.ravel()

    for ax, cid in zip(axes, cluster_ids):
        t0 = deploy_by_cluster[cid]
        t1 = t0 + pd.Timedelta(days=1)
        raw_c = raw[(raw["cluster"] == cid) & (raw["time"] >= t0) & (raw["time"] <= t1)]
        sm_c = smooth[(smooth["cluster"] == cid)
                      & (smooth["time"] >= t0) & (smooth["time"] <= t1)]

        ax.scatter(
            raw_c["lon"], raw_c["lat"], s=4, color="k", marker=",", alpha=0.25,
            transform=ccrs.PlateCarree(),
        )
        for dtype, type_sub in sm_c.groupby("type"):
            cmap = TYPE_CMAPS.get(dtype, plt.get_cmap("Greys"))
            dids = list(type_sub["drifter_id"].unique())
            for i, did in enumerate(dids):
                color = cmap(0.4 + 0.5 * (i / max(len(dids) - 1, 1)))
                sub = type_sub[type_sub["drifter_id"] == did]
                for _, seg in sub.groupby("segment_id"):
                    ax.plot(
                        seg["lon_smooth"], seg["lat_smooth"], color=color,
                        lw=1.2, marker="o", ms=2.5, transform=ccrs.PlateCarree(),
                    )

        # Frame on the preprocessed positions only, so raw outliers don't
        # blow up the extent (they are still drawn, just clipped).
        if not sm_c.empty:
            lons = sm_c["lon_smooth"].dropna()
            lats = sm_c["lat_smooth"].dropna()
            if len(lons) and len(lats):
                mx = max(0.05, 0.1 * (lons.max() - lons.min()))
                my = max(0.05, 0.1 * (lats.max() - lats.min()))
                ax.set_extent(
                    [lons.min() - mx, lons.max() + mx,
                     lats.min() - my, lats.max() + my],
                    crs=ccrs.PlateCarree(),
                )
        ax.set_title(f"{cid} — deploy {t0:%Y-%m-%d %H:%M}Z + 1 day")
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.COASTLINE, lw=0.5)
        ax.gridlines(draw_labels=True, alpha=0.3)
        types_here = sorted(sm_c["type"].unique())
        handles = [mlines.Line2D([], [], color="k", marker=".", linestyle="None",
                                 markersize=3, label="raw")]
        handles += [mlines.Line2D([], [], color=TYPE_CMAPS.get(t, plt.get_cmap("Greys"))(0.7),
                                  label=t) for t in types_here]
        ax.legend(handles=handles, loc="best", fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Per-cluster trajectories — raw fixes (gray) and smoothed tracks, "
                 "first day after deployment")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "trajectories_per_cluster.png", dpi=300)
    plt.close(fig)


def _spectrum(arr: np.ndarray, fs: float):
    arr = arr[np.isfinite(arr)]
    if len(arr) < 64:
        return None, None
    nperseg = min(512, len(arr))
    f, p = welch(arr, fs=fs, nperseg=nperseg, detrend="constant")
    return f, p


def _inertial_freq_per_hour(lat_deg: float) -> float:
    """Inertial frequency f = 2 Ω sin(lat), expressed in cycles per hour."""
    omega = 7.2921e-5  # Earth angular velocity, rad/s
    f_rad_per_s = 2.0 * omega * np.sin(np.deg2rad(abs(lat_deg)))
    return f_rad_per_s * 3600.0 / (2.0 * np.pi)


def spectra(smooth: pd.DataFrame, drifter_type: str):
    _ensure_dir()
    sub = smooth[smooth["type"] == drifter_type]
    if sub.empty:
        return
    fs = 1.0 / 3600.0  # samples per second (hourly)

    lats = sub["lat_smooth"].dropna().to_numpy()
    f_in_lo = _inertial_freq_per_hour(np.nanpercentile(lats, 5))
    f_in_hi = _inertial_freq_per_hour(np.nanpercentile(lats, 95))
    f_diurnal = 1.0 / 24.0
    f_m2 = 1.0 / 12.4206
    f_m4 = 1.0 / 6.2103

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for kind, ax, label in [
        (("vel_e_raw", "vel_n_raw", "vel_e_smooth", "vel_n_smooth"), axes[0], "velocity (m/s)^2/Hz"),
        (("acc_e_raw", "acc_n_raw", "acc_e_smooth", "acc_n_smooth"), axes[1], "acceleration (m/s^2)^2/Hz"),
    ]:
        e_raw, n_raw, e_sm, n_sm = kind
        # Stack across drifters: compute per-segment magnitude then concat.
        raw_mag = np.sqrt(sub[e_raw].to_numpy() ** 2 + sub[n_raw].to_numpy() ** 2)
        sm_mag = np.sqrt(sub[e_sm].to_numpy() ** 2 + sub[n_sm].to_numpy() ** 2)
        f_r, p_r = _spectrum(raw_mag, fs)
        f_s, p_s = _spectrum(sm_mag, fs)
        if f_r is not None:
            ax.loglog(f_r * 3600.0, p_r, label="raw (interp)")
        if f_s is not None:
            ax.loglog(f_s * 3600.0, p_s, label="smoothed")
        ax.axvspan(f_in_lo, f_in_hi, color="tab:red", alpha=0.15, label="inertial band")
        ax.axvline(f_diurnal, color="tab:green", lw=0.8, ls="--", label="diurnal")
        ax.axvline(f_m2, color="tab:purple", lw=0.8, ls=":", label="M2")
        ax.axvline(f_m4, color="tab:brown", lw=0.8, ls=":", label="M4")
        ax.set_xlabel("frequency (1/h)")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(f"{drifter_type}: power spectra")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"spectra_{drifter_type.lower()}.png", dpi=300)
    plt.close(fig)


def make_all(raw: pd.DataFrame, qc_df: pd.DataFrame, smooth: pd.DataFrame):
    for t in ("MARGET", "SPOT", "MELODI", "OMB"):
        counts_per_day(raw, smooth, t)
        sampling_histogram(raw, t)
        spectra(smooth, t)
    trajectories(raw, smooth)
    trajectories_per_cluster(raw, smooth)
