"""Load raw drifter JSON files, harmonize formats, and dedupe."""
import json
from pathlib import Path

import pandas as pd

from utils.geo import wrap_lon_180
from utils.io_paths import (
    DATA_RAW_MARGET,
    DATA_RAW_MELODI,
    DATA_RAW_OMB,
    DATA_RAW_SPOT,
)


def _load_json_dir(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not data:
            continue
        rows.extend(data)
    return rows


def _to_dataframe(rows: list[dict], drifter_type: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["type", "drifter_id", "time", "lat", "lon"]
        )
    df = pd.DataFrame(rows)
    df = df[["id", "dateTime", "latitude", "longitude"]].copy()
    df.columns = ["drifter_id", "time", "lat", "lon"]
    df["drifter_id"] = df["drifter_id"].astype(str)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed")
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    if drifter_type == "MARGET":
        df["lon"] = wrap_lon_180(df["lon"].to_numpy())
    df["type"] = drifter_type
    df = df.drop_duplicates(subset=["drifter_id", "time"], keep="first")
    df = df.sort_values(["drifter_id", "time"]).reset_index(drop=True)
    return df[["type", "drifter_id", "time", "lat", "lon"]]


def load_marget() -> pd.DataFrame:
    return _to_dataframe(_load_json_dir(DATA_RAW_MARGET), "MARGET")


def load_spot() -> pd.DataFrame:
    return _to_dataframe(_load_json_dir(DATA_RAW_SPOT), "SPOT")


def load_melodi() -> pd.DataFrame:
    return _to_dataframe(_load_json_dir(DATA_RAW_MELODI), "MELODI")


def load_omb() -> pd.DataFrame:
    return _to_dataframe(_load_json_dir(DATA_RAW_OMB), "OMB")


def load_all() -> pd.DataFrame:
    return pd.concat(
        [load_marget(), load_spot(), load_melodi(), load_omb()],
        ignore_index=True,
    )


if __name__ == "__main__":
    for name, loader in (
        ("MARGET", load_marget), ("SPOT", load_spot),
        ("MELODI", load_melodi), ("OMB", load_omb),
    ):
        d = loader()
        print(f"{name}: {len(d)} rows, {d['drifter_id'].nunique()} drifters")
