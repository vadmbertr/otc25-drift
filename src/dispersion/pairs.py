"""Build drifter pair separation series from the smoothed hourly tracks.

Pairs of drifters released together (same deploy time + lat/lon
in ``deploy_time.csv``).  Pair time origin = deployment time.

Returns a DataFrame with one row per (pair, lag) sample::

    pair_id, type_pair, drifter_a, drifter_b, cluster, t0, lag_h, r_km

``type_pair`` is ``SS`` for SPOT-SPOT, ``S-<TYPE>`` when exactly one drifter is
a SPOT (``S-MARGET``, ``S-MELODI``, ``S-OMB``), and ``<A>-<B>`` (sorted) for any
other non-SPOT combination.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from utils.geo import haversine_m
from utils.io_paths import DEPLOY_TIME_CSV


def _load_clusters() -> pd.DataFrame:
    """Group deploy_time.csv rows that share (deploy_time, lat, lon)."""
    df = pd.read_csv(DEPLOY_TIME_CSV, sep=r"\s+")
    df["deploy_time"] = pd.to_datetime(df["deploy_time"], utc=True, format="mixed")
    df["id"] = df["id"].astype(str)
    df["cluster"] = (
        df.groupby(["deploy_time", "lat", "lon"], sort=False).ngroup().add(1)
    )
    df["cluster"] = df["cluster"].map(lambda i: f"C{i:02d}")
    return df


def _drifter_timeline(df: pd.DataFrame, drifter_id: str) -> pd.DataFrame:
    """Per-drifter hourly track, indexed by time, with lon_smooth/lat_smooth."""
    sub = df.loc[df["drifter_id"] == drifter_id, ["time", "lat_smooth", "lon_smooth"]]
    sub = sub.dropna(subset=["lat_smooth", "lon_smooth"])
    sub = sub.drop_duplicates(subset=["time"]).sort_values("time")
    return sub.set_index("time")


def _pair_separation(
    track_a: pd.DataFrame,
    track_b: pd.DataFrame,
    t0: pd.Timestamp | None,
) -> pd.DataFrame:
    """Inner-join two tracks on time, return lag (h) and separation (km)."""
    joined = track_a.join(track_b, how="inner", lsuffix="_a", rsuffix="_b")
    if joined.empty:
        return joined
    if t0 is None:
        t0 = joined.index[0]
    lag_h = (joined.index - t0).total_seconds().to_numpy() / 3600.0
    r_m = haversine_m(
        joined["lon_smooth_a"].to_numpy(),
        joined["lat_smooth_a"].to_numpy(),
        joined["lon_smooth_b"].to_numpy(),
        joined["lat_smooth_b"].to_numpy(),
    )
    return pd.DataFrame(
        {"time": joined.index, "lag_h": lag_h, "r_km": r_m / 1000.0}
    )


def _pair_type(type_a: str, type_b: str) -> str:
    types = tuple(sorted((type_a, type_b)))
    if types == ("SPOT", "SPOT"):
        return "SS"
    if "SPOT" in types:
        other = types[1] if types[0] == "SPOT" else types[0]
        return f"S-{other}"
    return "-".join(types)


def build_pairs(drifters: pd.DataFrame) -> pd.DataFrame:
    """Within-cluster pairs.  t0 = deployment time of the cluster."""
    deploy = _load_clusters()
    type_by_id = (
        drifters.drop_duplicates("drifter_id").set_index("drifter_id")["type"].to_dict()
    )

    rows = []
    for cluster_id, cluster in deploy.groupby("cluster", sort=False):
        ids = list(cluster["id"])
        t0 = pd.Timestamp(cluster["deploy_time"].iloc[0])
        # Drop drifters that didn't make it through preprocessing.
        ids = [d for d in ids if d in type_by_id]
        if len(ids) < 2:
            continue
        tracks = {d: _drifter_timeline(drifters, d) for d in ids}
        for a, b in combinations(ids, 2):
            sep = _pair_separation(tracks[a], tracks[b], t0=t0)
            if sep.empty:
                continue
            sep = sep[sep["lag_h"] >= 0]
            if sep.empty:
                continue
            sep["pair_id"] = f"{cluster_id}:{a}__{b}"
            sep["drifter_a"] = a
            sep["drifter_b"] = b
            sep["type_pair"] = _pair_type(type_by_id[a], type_by_id[b])
            sep["cluster"] = cluster_id
            sep["t0"] = t0
            rows.append(sep)
    if not rows:
        return pd.DataFrame(
            columns=["pair_id", "type_pair", "drifter_a", "drifter_b",
                     "cluster", "t0", "time", "lag_h", "r_km"]
        )
    return pd.concat(rows, ignore_index=True)
