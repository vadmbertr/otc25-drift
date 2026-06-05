import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def wrap_lon_180(lon):
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


def lonlat_to_xy(lon, lat, lon0, lat0):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    cos_lat0 = np.cos(np.deg2rad(lat0))
    x = EARTH_RADIUS_M * np.deg2rad(lon - lon0) * cos_lat0
    y = EARTH_RADIUS_M * np.deg2rad(lat - lat0)
    return x, y


def xy_to_lonlat(x, y, lon0, lat0):
    cos_lat0 = np.cos(np.deg2rad(lat0))
    lon = lon0 + np.rad2deg(np.asarray(x, dtype=float) / (EARTH_RADIUS_M * cos_lat0))
    lat = lat0 + np.rad2deg(np.asarray(y, dtype=float) / EARTH_RADIUS_M)
    return lon, lat


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def to_posix_seconds(time_series) -> np.ndarray:
    """Convert a pandas datetime series (any tz/resolution) to POSIX seconds.

    Pandas 3 defaults to ``datetime64[us, UTC]`` for tz-aware columns built
    via ``pd.to_datetime(..., utc=True)``, so the older ``astype('int64')/1e9``
    idiom silently produces times that are 10**3 too small.  Going through
    UTC-stripped ``datetime64[ns]`` makes the conversion both
    resolution-independent and warning-free.
    """
    import pandas as pd
    dt = pd.to_datetime(time_series, utc=True)
    arr = dt.dt.tz_convert(None).to_numpy().astype("datetime64[ns]")
    return arr.astype("int64") / 1e9
