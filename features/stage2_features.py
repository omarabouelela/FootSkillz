import os
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------

PREPROC_DIR = r"C:\Users\omar1\Downloads\Footskillz\preprocessing"     # where stage 1 outputs are
FEATURES_DIR = r"C:\Users\omar1\Downloads\Footskillz\features"         # where to save feature CSV
METADATA_FILE = os.path.join(PREPROC_DIR, "windows_metadata.csv")

# if you ever change window size / overlap, features still work


# -----------------------------
# STATS HELPERS
# -----------------------------

def zero_crossing_rate(x: np.ndarray) -> float:
    """
    Zero-crossing rate: fraction of sign changes in the
    mean-removed signal.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0

    # remove DC component first
    xc = x - np.mean(x)

    signs = np.sign(xc)

    # propagate previous sign through zeros to avoid fake sign changes
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]

    prod = signs[1:] * signs[:-1]
    changes = np.sum(prod < 0)
    return float(changes) / float(len(x) - 1)


def series_features(x: np.ndarray, prefix: str) -> dict:
    """
    Compute features for one 1-D series.
    - mean
    - std
    - 2nd & 3rd central moments (mom2, mom3)
    - kurtosis (standardized 4th central moment)
    - zero-crossing rate
    """
    x = np.asarray(x, dtype=float)
    feats = {}

    if x.size == 0 or np.all(~np.isfinite(x)):
        # all NaN or empty
        feats[f"{prefix}_mean"] = np.nan
        feats[f"{prefix}_std"] = np.nan
        feats[f"{prefix}_mom2"] = np.nan
        feats[f"{prefix}_mom3"] = np.nan
        feats[f"{prefix}_kurtosis"] = np.nan
        feats[f"{prefix}_zcr"] = np.nan
        return feats

    x = np.nan_to_num(x, nan=np.nanmean(x))

    mu = np.mean(x)
    std = np.std(x, ddof=0)

    centered = x - mu
    m2 = np.mean(centered ** 2)           # 2nd central moment (variance)
    m3 = np.mean(centered ** 3)           # 3rd central moment
    m4 = np.mean(centered ** 4)           # 4th central moment

    eps = 1e-12
    if m2 > eps:
        kurt = m4 / (m2 ** 2 + eps) - 3.0  # excess kurtosis
    else:
        kurt = np.nan

    zcr = zero_crossing_rate(x)

    feats[f"{prefix}_mean"] = mu
    feats[f"{prefix}_std"] = std
    feats[f"{prefix}_mom2"] = m2
    feats[f"{prefix}_mom3"] = m3
    feats[f"{prefix}_kurtosis"] = kurt
    feats[f"{prefix}_zcr"] = zcr

    return feats


def magnitude_from_df(df: pd.DataFrame) -> np.ndarray:
    """Compute vector magnitude sqrt(x^2 + y^2 + z^2) from window dataframe."""
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    return np.sqrt(x * x + y * y + z * z)


# -----------------------------
# WINDOW FEATURE EXTRACTION
# -----------------------------

def features_for_window(preproc_root: Path,
                        row: pd.Series) -> dict:
    """
    Given one row from windows_metadata, load its acc/gyro/grav
    window CSVs and compute features.
    """
    skill = row["skill"]
    recording = row["recording"]
    window_id = int(row["window_id"])

    base_dir = preproc_root / skill / recording
    base_name = f"win_{window_id:03d}"

    acc_path = base_dir / f"{base_name}_acc.csv"
    gyro_path = base_dir / f"{base_name}_gyro.csv"
    grav_path = base_dir / f"{base_name}_grav.csv"

    # read windows
    acc_df = pd.read_csv(acc_path)
    gyro_df = pd.read_csv(gyro_path)
    grav_df = pd.read_csv(grav_path)

    # base info
    feat = {
        "skill": skill,
        "recording": recording,
        "window_id": window_id,
        "start_time": float(row["start_time"]),
        "end_time": float(row["end_time"]),
        "n_samples": int(row["n_samples"]),
        "fs": float(row["fs"]),
        "pause_time": float(row["pause_time"]),
    }

    # for each sensor: x, y, z and magnitude
    sensors = {
        "acc": acc_df,
        "gyro": gyro_df,
        "grav": grav_df,
    }

    for s_name, df in sensors.items():
        # axis features
        for axis in ["x", "y", "z"]:
            if axis not in df.columns:
                continue
            x = df[axis].to_numpy(dtype=float)
            prefix = f"{s_name}_{axis}"
            feat.update(series_features(x, prefix))

        # magnitude features
        mag = magnitude_from_df(df)
        prefix_mag = f"{s_name}_mag"
        feat.update(series_features(mag, prefix_mag))

    return feat


# -----------------------------
# MAIN
# -----------------------------

def main():
    preproc_root = Path(PREPROC_DIR)
    features_root = Path(FEATURES_DIR)
    features_root.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(METADATA_FILE)

    all_feats = []
    total = len(meta)

    for idx, row in meta.iterrows():
        feat_row = features_for_window(preproc_root, row)
        all_feats.append(feat_row)

        # optional: tiny progress print
        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"Processed {idx + 1}/{total} windows")

    feats_df = pd.DataFrame(all_feats)
    feats_df.sort_values(["skill", "recording", "window_id"], inplace=True)

    out_path = features_root / "features_all_windows.csv"
    feats_df.to_csv(out_path, index=False)
    print(f"\nSaved features to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
