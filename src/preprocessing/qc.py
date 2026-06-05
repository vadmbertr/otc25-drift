"""Quality control: pre-deployment removal, kinematic + statistical outlier filters, segmentation."""
import numpy as np
import pandas as pd

from utils.geo import haversine_m, lonlat_to_xy, to_posix_seconds
from utils.io_paths import DEPLOY_TIME_CSV


WINDOW = 11               # number of recent validated samples used to build the MAD scale
SIGMA_K = 5.0
GAP_THRESHOLD = pd.Timedelta(hours=6)
V_MAX = 2.0               # m/s; max plausible drifter speed
# Floor on the MAD-derived implied-acceleration scale (m/s^2).  Demol 2025
# puts the drifter acceleration prior std at sigma_acc = 4.89e-6 m/s^2; the
# floor at a few times that keeps the threshold meaningful when the MAD
# history is empty or dominated by tiny accelerations.
ACC_SCALE_FLOOR = 2.0e-5


def load_deploy_times(path=DEPLOY_TIME_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+")
    df["deploy_time"] = pd.to_datetime(df["deploy_time"], utc=True, format="mixed")
    df["id"] = df["id"].astype(str)
    return df


def remove_pre_deployment(df: pd.DataFrame, deploy: pd.DataFrame) -> pd.DataFrame:
    deploy_map = dict(zip(deploy["id"], deploy["deploy_time"]))
    keep = []
    for drifter_id, sub in df.groupby("drifter_id", sort=False):
        dt = deploy_map.get(drifter_id)
        if dt is None:
            keep.append(sub)
        else:
            keep.append(sub[sub["time"] >= dt])
    if not keep:
        return df.iloc[0:0]
    return pd.concat(keep, ignore_index=True)


def filter_speed_one(sub: pd.DataFrame, v_max: float = V_MAX) -> pd.DataFrame:
    sub = sub.sort_values("time").reset_index(drop=True)
    n = len(sub)
    if n < 2:
        return sub
    lat = sub["lat"].to_numpy()
    lon = sub["lon"].to_numpy()
    t = sub["time"].to_numpy()
    keep = [0]
    anchor = 0
    for i in range(1, n):
        dt_s = (t[i] - t[anchor]) / np.timedelta64(1, "s")
        if dt_s <= 0:
            raise ValueError(
                f"Non-positive time difference between fixes {anchor} and {i} in drifter {sub['drifter_id'].iloc[0]}"
            )
        d = haversine_m(lon[anchor], lat[anchor], lon[i], lat[i])
        if d / dt_s <= v_max:
            keep.append(i)
            anchor = i
    return sub.iloc[keep].reset_index(drop=True)


def filter_speed(df: pd.DataFrame) -> pd.DataFrame:
    parts = [filter_speed_one(sub) for _, sub in df.groupby("drifter_id", sort=False)]
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def filter_outliers_one(sub: pd.DataFrame) -> pd.DataFrame:
    """Sequential 3-point implied-acceleration filter.

    Walks through the fixes in chronological order.  At each interior
    candidate ``i`` the predicted position is the time-linear
    interpolation between the last validated fix and the next candidate
    in the original stream; the implied 3-point acceleration is

        a_i = 2 * || x_i - x_pred || / (dt_left * dt_right)

    in m/s^2, where ``dt_left = t_i - t_{last_valid}`` and
    ``dt_right = t_{next} - t_i``.  Under a local parabolic trajectory
    model this is the magnitude of the constant acceleration that
    would explain the observed deviation from the chord, so the same
    drifter physics produces the same ``a_i`` regardless of sampling
    rate -- the test is gap-invariant where the raw residual was not.

    The threshold is ``k * 1.4826 * MAD`` of the most recent ``WINDOW``
    validated ``a_i`` values, floored at ``ACC_SCALE_FLOOR`` to keep
    the threshold sane when the MAD history is empty (start of
    trajectory) or dominated by very small accelerations.  A candidate
    whose ``a_i`` exceeds the threshold is dropped, ``last_valid`` is
    *not* advanced, and the scan moves to the next candidate -- so a
    bad fix only ever evicts itself, never its good neighbours.  One
    sweep through the trajectory is enough; the MAD heals after every
    removal because the next decision uses only validated history.
    """
    sub = sub.sort_values("time").reset_index(drop=True)
    n = len(sub)
    if n < 3:
        return sub

    lat = sub["lat"].to_numpy()
    lon = sub["lon"].to_numpy()
    t_s = to_posix_seconds(sub["time"])

    lat0 = float(np.mean(lat))
    lon0 = float(np.mean(lon))
    x, y = lonlat_to_xy(lon, lat, lon0, lat0)

    keep = np.ones(n, dtype=bool)
    valid_a: list[float] = []
    last_valid = 0  # first fix always accepted; endpoints already vetted by V_MAX

    for i in range(1, n - 1):
        nxt = i + 1
        dt_left = t_s[i] - t_s[last_valid]
        dt_right = t_s[nxt] - t_s[i]
        if dt_left <= 0 or dt_right <= 0:
            keep[i] = False
            continue
        alpha = dt_left / (dt_left + dt_right)
        x_pred = x[last_valid] + alpha * (x[nxt] - x[last_valid])
        y_pred = y[last_valid] + alpha * (y[nxt] - y[last_valid])
        res = float(np.hypot(x[i] - x_pred, y[i] - y_pred))
        a_imp = 2.0 * res / (dt_left * dt_right)  # m/s^2

        if len(valid_a) >= 3:
            recent = np.asarray(valid_a[-WINDOW:])
            mad = float(np.median(np.abs(recent - np.median(recent))))
            scale = max(1.4826 * mad, ACC_SCALE_FLOOR)
        else:
            scale = ACC_SCALE_FLOOR

        if a_imp > SIGMA_K * scale:
            keep[i] = False
            # do NOT advance last_valid: the next iteration uses the
            # same anchor through fix i+1, closing the gap.
        else:
            valid_a.append(a_imp)
            last_valid = i

    return sub.iloc[keep].reset_index(drop=True)


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    parts = [filter_outliers_one(sub) for _, sub in df.groupby("drifter_id", sort=False)]
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def assign_segments(df: pd.DataFrame, gap_threshold=GAP_THRESHOLD) -> pd.DataFrame:
    out = []
    for drifter_id, sub in df.groupby("drifter_id", sort=False):
        sub = sub.sort_values("time").reset_index(drop=True)
        dt = sub["time"].diff()
        new_seg = (dt > gap_threshold).fillna(False).to_numpy()
        seg_idx = np.cumsum(new_seg)
        sub = sub.copy()
        sub["segment_id"] = [f"{drifter_id}_{i:03d}" for i in seg_idx]
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def run_qc(raw: pd.DataFrame) -> pd.DataFrame:
    deploy = load_deploy_times()
    after_deploy = remove_pre_deployment(raw, deploy)
    after_speed = filter_speed(after_deploy)
    filtered = filter_outliers(after_speed)
    return assign_segments(filtered)
