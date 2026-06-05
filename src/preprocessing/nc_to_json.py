"""Export selected NetCDF drifter trajectories to the raw-JSON format.

`data/OTC25_drifters_trajectories.nc` carries MELODI / OMB / NANO-OMB platforms
that are not part of the JSON pipeline.  Four of them are manually assigned to
existing deployment clusters; this script writes their observations to the same
raw-JSON schema as SPOT (``{id, dateTime, latitude, longitude}``), one file per
UTC day, under ``data/raw/melodi`` and ``data/raw/omb`` so the standard loader
picks them up.

Run once:  ``python src/preprocessing/nc_to_json.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io_paths import DATA_RAW_MELODI, DATA_RAW_OMB  # noqa: E402

NC_PATH = Path(__file__).resolve().parents[2] / "data" / "OTC25_drifters_trajectories.nc"

# nc trajectory name -> output directory (only the cluster-assigned drifters).
ASSIGNED = {
    "OTC25-MELODI-15": DATA_RAW_MELODI,
    "OTC25-MELODI-04": DATA_RAW_MELODI,
    "OTC25-MELODI-14": DATA_RAW_MELODI,
    "OTC25-OMB-07": DATA_RAW_OMB,
}


def _trajectory_records(ds: xr.Dataset) -> dict[str, list[dict]]:
    """Per assigned trajectory, build SPOT-style records sorted by time."""
    names = list(ds["trajectory"].values)
    rowsize = ds["rowSize"].values.astype(int)
    offsets = np.concatenate([[0], np.cumsum(rowsize)])
    time = pd.to_datetime(ds["time"].values, utc=True)
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    out: dict[str, list[dict]] = {}
    for name in ASSIGNED:
        i = names.index(name)
        s, e = int(offsets[i]), int(offsets[i + 1])
        t, la, lo = time[s:e], lat[s:e], lon[s:e]
        ok = np.isfinite(la) & np.isfinite(lo)
        recs = [
            {
                "id": name,
                "dateTime": ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "latitude": float(a),
                "longitude": float(o),
            }
            for ts, a, o in zip(t[ok], la[ok], lo[ok])
        ]
        out[name] = recs
    return out


def main() -> None:
    ds = xr.open_dataset(NC_PATH)
    records = _trajectory_records(ds)

    # Group all records destined for one directory by UTC calendar day.
    by_dir: dict[Path, list[dict]] = {DATA_RAW_MELODI: [], DATA_RAW_OMB: []}
    for name, recs in records.items():
        by_dir[ASSIGNED[name]].extend(recs)

    for directory, recs in by_dir.items():
        directory.mkdir(parents=True, exist_ok=True)
        if not recs:
            continue
        df = pd.DataFrame(recs)
        day = pd.to_datetime(df["dateTime"], utc=True).dt.floor("1D")
        n_files = 0
        for d, grp in df.groupby(day):
            fname = d.strftime("%Y-%m-%dT%H:%M:%SZ.json")
            payload = grp.drop(columns=[]).to_dict(orient="records")
            (directory / fname).write_text(json.dumps(payload, indent=1))
            n_files += 1
        ids = sorted(df["id"].unique())
        print(f"{directory.name}: {len(recs)} obs, {len(ids)} drifters {ids}, "
              f"{n_files} daily files")


if __name__ == "__main__":
    main()
