import os
import io
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------

READINGS_ZIP = r"C:\Users\omar1\Downloads\readings.zip"      # the outer zip you gave me
OUTPUT_DIR = "processed_stage1"     # where cleaned windows will be saved

WINDOW_SEC = 2.0                    # window length (s)
STEP_SEC = 1.0                      # step between windows (s) -> 50% overlap
PAUSE_RMS_WINDOW_SEC = 1.0          # smoothing window for pause detection (s)
PAUSE_THRESHOLD_MIN = 0.05          # minimum RMS threshold for pause detection
SPIKE_Z_THRESH = 8.0                # robust z-score threshold for spikes


# -----------------------------
# Helper functions
# -----------------------------

def robust_z(x, eps: float = 1e-9) -> np.ndarray:
    """Robust z-score using median and MAD."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + eps
    return 0.6745 * (x - med) / mad


def rolling_rms(x: np.ndarray, fs: float, window_sec: float) -> np.ndarray:
    """Rolling RMS over a given window size in seconds."""
    x = np.asarray(x, dtype=float)
    win = max(int(round(window_sec * fs)), 1)
    s = pd.Series(x).rolling(win, min_periods=max(3, win // 4)).mean()
    return np.sqrt(s.to_numpy())


def load_sensor_csv_from_zip(zf: zipfile.ZipFile, member_name: str) -> pd.DataFrame:
    """Load a CSV from an inner zip and standardise time column."""
    with zf.open(member_name) as f:
        df = pd.read_csv(f)

    # normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # use seconds_elapsed if present, otherwise time
    if "seconds_elapsed" in df.columns:
        t = pd.to_numeric(df["seconds_elapsed"], errors="coerce")
    elif "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce")
    else:
        raise ValueError(f"No time / seconds_elapsed column in {member_name}")

    # ensure float and start at 0
    t = t.astype(float)
    t = t - t.iloc[0]
    df["t"] = t

    return df


def magnitude(df: pd.DataFrame) -> np.ndarray:
    """Compute vector magnitude from x, y, z columns."""
    cols = df.columns
    if {"x", "y", "z"}.issubset(cols):
        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        z = df["z"].to_numpy(dtype=float)
    else:
        # fall back to any *_x, *_y, *_z
        x_col = next(c for c in cols if c.endswith("x"))
        y_col = next(c for c in cols if c.endswith("y"))
        z_col = next(c for c in cols if c.endswith("z"))
        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        z = df[z_col].to_numpy(dtype=float)

    return np.sqrt(x * x + y * y + z * z)


def estimate_fs(t: np.ndarray) -> float:
    """Estimate sampling frequency from time vector."""
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Cannot estimate sampling rate (no valid dt).")
    med_dt = np.median(dt)
    return 1.0 / med_dt


def detect_pause(acc_mag: np.ndarray,
                 t: np.ndarray,
                 fs: float):
    """
    Detect pause region using RMS of accelerometer magnitude.
    Returns onset index, onset time, threshold and RMS vector.
    """
    rms = rolling_rms(acc_mag, fs, PAUSE_RMS_WINDOW_SEC)
    rms = np.asarray(rms, dtype=float)
    good = np.isfinite(rms)
    if not good.any():
        return 0, float(t[0]), PAUSE_THRESHOLD_MIN, rms

    base = rms[good]
    thr = np.nanpercentile(base, 25) + 0.25 * np.nanstd(base)
    thr = float(max(thr, PAUSE_THRESHOLD_MIN))

    idx = np.where(rms > thr)[0]
    if idx.size == 0:
        onset_idx = 0
    else:
        onset_idx = int(idx[0])

    onset_time = float(t[onset_idx])
    return onset_idx, onset_time, thr, rms


def clean_spikes(x: np.ndarray) -> np.ndarray:
    """
    Replace spikes (|robust z| > SPIKE_Z_THRESH) with local median.
    """
    x = np.asarray(x, dtype=float)
    z = robust_z(x)
    spikes = np.abs(z) > SPIKE_Z_THRESH
    if not spikes.any():
        return x

    x_clean = x.copy()
    spike_idx = np.where(spikes)[0]

    for i in spike_idx:
        left = max(0, i - 2)
        right = min(len(x), i + 3)
        neighbourhood = x[left:right]
        neigh_mask = ~spikes[left:right]
        neigh_vals = neighbourhood[neigh_mask]
        if neigh_vals.size > 0:
            x_clean[i] = np.median(neigh_vals)
        else:
            x_clean[i] = np.median(x[~spikes])

    return x_clean


def window_indices(n: int,
                   fs: float,
                   window_sec: float,
                   step_sec: float):
    """Compute (start, end) sample indices for overlapping windows."""
    win = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    if win <= 0 or step <= 0:
        raise ValueError("Invalid window/step size.")
    if n < win:
        return []

    starts = list(range(0, n - win + 1, step))
    return [(s, s + win) for s in starts]


# -----------------------------
# Core processing per recording
# -----------------------------

def process_one_recording(zip_bytes: bytes,
                          recording_name: str,
                          skill_label: str,
                          out_dir: Path):

    # open the inner recording zip
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as reczip:
        members = reczip.namelist()

        def find_member(keyword: str) -> str:
            for m in members:
                if keyword.lower() in m.lower() and m.lower().endswith(".csv"):
                    return m
            raise FileNotFoundError(f"No CSV containing '{keyword}' in {recording_name}")

        acc_member = find_member("accelerometer")
        gyro_member = find_member("gyroscope")
        grav_member = find_member("gravity")

        acc_df = load_sensor_csv_from_zip(reczip, acc_member)
        gyro_df = load_sensor_csv_from_zip(reczip, gyro_member)
        grav_df = load_sensor_csv_from_zip(reczip, grav_member)

    # time & sampling from accelerometer
    t_acc = acc_df["t"].to_numpy(dtype=float)
    fs = estimate_fs(t_acc)

    # magnitudes for pause & spikes
    acc_mag = magnitude(acc_df)
    gyro_mag = magnitude(gyro_df)
    grav_mag = magnitude(grav_df)

    acc_mag_clean = clean_spikes(acc_mag)
    gyro_mag_clean = clean_spikes(gyro_mag)
    grav_mag_clean = clean_spikes(grav_mag)  # not strictly needed for pause

    # pause detection using cleaned accelerometer magnitude
    onset_idx, onset_time, thr, rms = detect_pause(acc_mag_clean, t_acc, fs)

    # ---- FIXED PART: trim each sensor separately and align lengths ----

    def trim_df_generic(df: pd.DataFrame, onset_index: int) -> pd.DataFrame:
        df_post = df.iloc[onset_index:].copy().reset_index(drop=True)
        if "t" in df_post.columns:
            df_post["t"] = df_post["t"] - df_post["t"].iloc[0]
        return df_post

    acc_post = trim_df_generic(acc_df, onset_idx)
    gyro_post = trim_df_generic(gyro_df, onset_idx)
    grav_post = trim_df_generic(grav_df, onset_idx)

    # make sure all have the same length (cut to the shortest)
    n_min = min(len(acc_post), len(gyro_post), len(grav_post))
    acc_post = acc_post.iloc[:n_min].reset_index(drop=True)
    gyro_post = gyro_post.iloc[:n_min].reset_index(drop=True)
    grav_post = grav_post.iloc[:n_min].reset_index(drop=True)

    t_post = acc_post["t"].to_numpy(dtype=float)
    n = len(t_post)

    # -------------------------------------------------------------------

    win_pairs = window_indices(n, fs, WINDOW_SEC, STEP_SEC)

    rec_base = Path(recording_name).stem
    rec_dir = out_dir / skill_label / rec_base
    rec_dir.mkdir(parents=True, exist_ok=True)

    windows_meta = []

    for w_idx, (start, end) in enumerate(win_pairs):
        # slice windows
        acc_w = acc_post.iloc[start:end].copy()
        gyro_w = gyro_post.iloc[start:end].copy()
        grav_w = grav_post.iloc[start:end].copy()

        # spike cleaning on each axis within each window
        for df_w in (acc_w, gyro_w, grav_w):
            for axis in ["x", "y", "z"]:
                if axis in df_w.columns:
                    df_w[axis] = clean_spikes(df_w[axis].to_numpy())

        w_name = f"win_{w_idx:03d}"
        acc_w.to_csv(rec_dir / f"{w_name}_acc.csv", index=False)
        gyro_w.to_csv(rec_dir / f"{w_name}_gyro.csv", index=False)
        grav_w.to_csv(rec_dir / f"{w_name}_grav.csv", index=False)

        windows_meta.append({
            "skill": skill_label,
            "recording": rec_base,
            "window_id": w_idx,
            "start_time": float(t_post[start]),
            "end_time": float(t_post[end - 1]),
            "n_samples": int(end - start),
            "fs": float(fs),
            "pause_time": float(onset_time),
            "pause_index": int(onset_idx),
            "threshold_rms": float(thr),
        })

    return windows_meta


# -----------------------------
# Top-level processing
# -----------------------------

def process_readings_zip(readings_zip_path: str,
                         out_dir: str = OUTPUT_DIR):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_meta = []

    with zipfile.ZipFile(readings_zip_path, "r") as outer:
        for member in outer.infolist():
            if member.is_dir():
                continue
            if not member.filename.lower().endswith(".zip"):
                # skip any non-inner zip files
                continue

            inner_path = member.filename
            skill_label = Path(inner_path).parent.name   # Heading / Pass / ...
            recording_name = Path(inner_path).name

            zip_bytes = outer.read(inner_path)

            meta = process_one_recording(
                zip_bytes=zip_bytes,
                recording_name=recording_name,
                skill_label=skill_label,
                out_dir=out_dir,
            )
            all_meta.extend(meta)

    meta_df = pd.DataFrame(all_meta)
    meta_df.sort_values(["skill", "recording", "window_id"], inplace=True)
    meta_df.to_csv(out_dir / "windows_metadata.csv", index=False)
    print(f"Done. Saved cleaned windows + metadata to: {out_dir.resolve()}")


if __name__ == "__main__":
    process_readings_zip(READINGS_ZIP, OUTPUT_DIR)
