"""LaTeX statistic tables for the preprocessing step."""
import numpy as np
import pandas as pd

from utils.geo import haversine_m
from utils.io_paths import STATS_DIR


def _ensure_dir():
    STATS_DIR.mkdir(parents=True, exist_ok=True)


def total_counts(raw: pd.DataFrame, smooth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in ("MARGET", "SPOT", "MELODI", "OMB"):
        rows.append(
            {
                "type": t,
                "raw_obs": int((raw["type"] == t).sum()),
                "filtered_obs": int((smooth["type"] == t).sum()),
                "drifters": int(raw.loc[raw["type"] == t, "drifter_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def travel_distance(smooth: pd.DataFrame) -> pd.DataFrame:
    """Per-segment and per-drifter cumulative great-circle distance (km), on raw and smoothed positions."""
    out_seg = []
    for (t, drifter_id, seg_id), sub in smooth.groupby(
        ["type", "drifter_id", "segment_id"], sort=False
    ):
        sub = sub.dropna(subset=["lat_smooth", "lon_smooth", "lat_raw", "lon_raw"])
        if len(sub) < 2:
            d_smooth = 0.0
            d_raw = 0.0
        else:
            lon_smooth = sub["lon_smooth"].to_numpy()
            lat_smooth = sub["lat_smooth"].to_numpy()
            lon_raw = sub["lon_raw"].to_numpy()
            lat_raw = sub["lat_raw"].to_numpy()
            d_smooth = float(np.sum(
                haversine_m(lon_smooth[:-1], lat_smooth[:-1], lon_smooth[1:], lat_smooth[1:]) / 1000.0
            ))
            d_raw = float(np.sum(haversine_m(lon_raw[:-1], lat_raw[:-1], lon_raw[1:], lat_raw[1:]) / 1000.0))
        out_seg.append(
            {
                "type": t, "drifter_id": drifter_id, "segment_id": seg_id, 
                "raw_distance_km": d_raw, "smoothed_distance_km": d_smooth
            }
        )
    seg = pd.DataFrame(out_seg)
    drifter = (
        seg.groupby(["type", "drifter_id"], as_index=False)[["raw_distance_km", "smoothed_distance_km"]].sum()
    )
    return seg, drifter


def operation_time(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (t, drifter_id), sub in raw.groupby(["type", "drifter_id"], sort=False):
        span = sub["time"].max() - sub["time"].min()
        rows.append(
            {
                "type": t,
                "drifter_id": drifter_id,
                "first_fix": sub["time"].min(),
                "last_fix": sub["time"].max(),
                "operation_hours": span.total_seconds() / 3600.0,
            }
        )
    return pd.DataFrame(rows)


def make_all(raw: pd.DataFrame, smooth: pd.DataFrame):
    _ensure_dir()

    counts = total_counts(raw, smooth)
    counts.to_latex(STATS_DIR / "counts.tex", index=False, float_format="%.0f")

    seg, drifter = travel_distance(smooth)
    with (STATS_DIR / "travel_distance.tex").open("w") as f:
        f.write("% Per-drifter total travel distance (raw + smoothed, km)\n")
        f.write(drifter.to_latex(index=False, float_format="%.2f"))
        f.write("\n% Per-segment travel distance (raw + smoothed, km)\n")
        f.write(seg.to_latex(index=False, float_format="%.2f"))

    op = operation_time(raw)
    op_out = op.copy()
    op_out["first_fix"] = op_out["first_fix"].dt.strftime("%Y-%m-%d %H:%M")
    op_out["last_fix"] = op_out["last_fix"].dt.strftime("%Y-%m-%d %H:%M")
    op_out.to_latex(STATS_DIR / "operation_time.tex", index=False, float_format="%.1f")
