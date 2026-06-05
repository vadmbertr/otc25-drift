"""Variational smoothing onto an hourly grid + finite-difference derivatives.

Model (Demol 2025, eq. 3.5), per coordinate component (E, N) in meters:

    minimize  J(s) = (1/sigma**2) || y - H s ||**2  +  (D2 s)^T R^{-1} (D2 s)

where s is on a regular hourly grid, H linearly interpolates the grid to the
raw fix times, D2 is the second-difference operator, and R is an exponential
acceleration autocovariance with R^{-1} closed-form tridiagonal.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from utils.geo import lonlat_to_xy, to_posix_seconds, xy_to_lonlat


# Demol 2025 recommended parameters.
SIGMA_POS = 37.0          # m, position noise std
SIGMA_ACC = 4.89e-6       # m / s**2, acceleration autocovariance amplitude
TAU_ACC_SEC = 4.0 * 3600  # s, acceleration decorrelation timescale
DT = 3600.0               # s, hourly grid spacing

MIN_SEG_OBS = 5


def _exp_inv_tridiag(n: int, r: float) -> csr_matrix:
    """Closed-form inverse of the n x n exponential covariance K_ij = r**|i-j|.

    The inverse is tridiagonal:
        diagonal: 1/(1-r**2) at corners, (1+r**2)/(1-r**2) in the interior
        off-diagonal: -r/(1-r**2)
    """
    if n <= 0:
        return csr_matrix((0, 0))
    denom = 1.0 - r * r
    diag = np.full(n, (1.0 + r * r) / denom)
    diag[0] = 1.0 / denom
    diag[-1] = 1.0 / denom
    off = np.full(n - 1, -r / denom)
    return diags([off, diag, off], offsets=[-1, 0, 1], format="csr")


def _second_diff_matrix(n: int, dt: float) -> csr_matrix:
    """(n-2) x n second-difference operator: row i is [..1, -2, 1..] / dt**2."""
    if n < 3:
        return csr_matrix((0, n))
    rows = np.repeat(np.arange(n - 2), 3)
    cols = rows + np.tile(np.array([0, 1, 2]), n - 2)
    data = np.tile(np.array([1.0, -2.0, 1.0]) / (dt * dt), n - 2)
    return csr_matrix((data, (rows, cols)), shape=(n - 2, n))


def _interp_matrix(t_obs: np.ndarray, t_grid: np.ndarray) -> csr_matrix:
    """Sparse linear-interpolation operator H (m x n)."""
    n = len(t_grid)
    m = len(t_obs)
    if m == 0:
        return csr_matrix((0, n))
    idx = np.searchsorted(t_grid, t_obs, side="right") - 1
    idx = np.clip(idx, 0, n - 2)
    dt_local = t_grid[idx + 1] - t_grid[idx]
    alpha = (t_obs - t_grid[idx]) / dt_local
    alpha = np.clip(alpha, 0.0, 1.0)
    rows = np.repeat(np.arange(m), 2)
    cols = np.empty(2 * m, dtype=int)
    cols[0::2] = idx
    cols[1::2] = idx + 1
    data = np.empty(2 * m)
    data[0::2] = 1.0 - alpha
    data[1::2] = alpha
    return csr_matrix((data, (rows, cols)), shape=(m, n))


def _smooth_component(
    y: np.ndarray,
    H: csr_matrix,
    D2: csr_matrix,
    Rinv: csr_matrix,
    sigma: float,
) -> np.ndarray:
    A = (H.T @ H) / (sigma * sigma) + D2.T @ Rinv @ D2
    b = (H.T @ y) / (sigma * sigma)
    return spsolve(A.tocsc(), b)


def _central_diff(s: np.ndarray, dt: float, order: int) -> np.ndarray:
    out = np.full_like(s, np.nan, dtype=float)
    if len(s) < 3:
        return out
    if order == 1:
        out[1:-1] = (s[2:] - s[:-2]) / (2.0 * dt)
    elif order == 2:
        out[1:-1] = (s[2:] - 2.0 * s[1:-1] + s[:-2]) / (dt * dt)
    else:
        raise ValueError(order)
    return out


def smooth_segment(
    sub: pd.DataFrame,
    sigma: float = SIGMA_POS,
    sigma_acc: float = SIGMA_ACC,
    tau_acc: float = TAU_ACC_SEC,
    dt: float = DT,
) -> pd.DataFrame | None:
    sub = sub.sort_values("time").reset_index(drop=True)
    if len(sub) < MIN_SEG_OBS:
        return None

    t_obs = to_posix_seconds(sub["time"])  # POSIX seconds
    lat0 = float(sub["lat"].mean())
    lon0 = float(sub["lon"].mean())
    x_obs, y_obs = lonlat_to_xy(sub["lon"].to_numpy(), sub["lat"].to_numpy(), lon0, lat0)

    t_start = pd.Timestamp(sub["time"].iloc[0]).floor("1h")
    t_end = pd.Timestamp(sub["time"].iloc[-1]).ceil("1h")
    t_grid_pd = pd.date_range(t_start, t_end, freq="1h", tz="UTC")
    n = len(t_grid_pd)
    if n < 3:
        return None
    t_grid = to_posix_seconds(pd.Series(t_grid_pd))

    # Raw (linearly interpolated onto hourly grid) - keep NaN outside obs range.
    in_range = (t_grid >= t_obs[0]) & (t_grid <= t_obs[-1])
    x_raw = np.full(n, np.nan)
    y_raw = np.full(n, np.nan)
    x_raw[in_range] = np.interp(t_grid[in_range], t_obs, x_obs)
    y_raw[in_range] = np.interp(t_grid[in_range], t_obs, y_obs)

    # Variational smoother.
    H = _interp_matrix(t_obs, t_grid)
    D2 = _second_diff_matrix(n, dt)
    r_corr = float(np.exp(-dt / tau_acc))
    # R = sigma_acc**2 * K with K_ij = r_corr**|i-j| over the (n-2)-long acceleration series.
    Kinv = _exp_inv_tridiag(n - 2, r_corr)
    Rinv = Kinv / (sigma_acc * sigma_acc)
    x_sm = _smooth_component(x_obs, H, D2, Rinv, sigma)
    y_sm = _smooth_component(y_obs, H, D2, Rinv, sigma)

    lon_raw_grid, lat_raw_grid = xy_to_lonlat(x_raw, y_raw, lon0, lat0)
    lon_sm_grid, lat_sm_grid = xy_to_lonlat(x_sm, y_sm, lon0, lat0)

    # Finite-difference derivatives in m/s and m/s**2 (E and N components).
    vel_e_raw = _central_diff(x_raw, dt, 1)
    vel_n_raw = _central_diff(y_raw, dt, 1)
    acc_e_raw = _central_diff(x_raw, dt, 2)
    acc_n_raw = _central_diff(y_raw, dt, 2)
    vel_e_sm = _central_diff(x_sm, dt, 1)
    vel_n_sm = _central_diff(y_sm, dt, 1)
    acc_e_sm = _central_diff(x_sm, dt, 2)
    acc_n_sm = _central_diff(y_sm, dt, 2)

    out = pd.DataFrame(
        {
            "type": sub["type"].iloc[0],
            "drifter_id": sub["drifter_id"].iloc[0],
            "segment_id": sub["segment_id"].iloc[0],
            "time": t_grid_pd,
            "lat_raw": lat_raw_grid,
            "lon_raw": lon_raw_grid,
            "lat_smooth": lat_sm_grid,
            "lon_smooth": lon_sm_grid,
            "vel_e_raw": vel_e_raw,
            "vel_n_raw": vel_n_raw,
            "vel_e_smooth": vel_e_sm,
            "vel_n_smooth": vel_n_sm,
            "acc_e_raw": acc_e_raw,
            "acc_n_raw": acc_n_raw,
            "acc_e_smooth": acc_e_sm,
            "acc_n_smooth": acc_n_sm,
        }
    )
    return out


def smooth_all(qc_df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for seg_id, sub in qc_df.groupby("segment_id", sort=False):
        out = smooth_segment(sub)
        if out is not None:
            pieces.append(out)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)
