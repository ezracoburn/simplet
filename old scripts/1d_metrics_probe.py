#!/usr/bin/env python3
import os
import re
import glob
import csv
from datetime import datetime

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.signal import peak_prominences

# --------------------------------------------------
# USER SETTINGS (EDIT THESE)
# --------------------------------------------------

GOOD_DIR = "/Users/ezracoburn/Documents/Simple/output/examples for 1D/good examples"
BAD_DIR = "/Users/ezracoburn/Documents/Simple/output/examples for 1D/problematic examples"

OUTPUT_DIR = "/Users/ezracoburn/Documents/Simple/output/1d_metrics"

# Metric params (keep aligned with your main pipeline)
TEX_BINS = 256
TEX_WIN = 9
HIST_SMOOTH_K = 9
ANGLE_DEG = 1.0
ANGLE_RUN = 10
MIN_C = -50.0
MAX_C = 200.0

# Optional: if True, recurse through subfolders inside GOOD_DIR / BAD_DIR
RECURSIVE = False

# --------------------------------------------------
# Discovery
# --------------------------------------------------

IRX_RE = re.compile(r"IRX_(\d{4})", re.IGNORECASE)

def find_frame_id(path: str):
    m = IRX_RE.search(os.path.basename(path))
    return m.group(1) if m else None

def list_irx_tiffs(root: str):
    if RECURSIVE:
        patterns = [
            "**/IRX_*.tif", "**/IRX_*.tiff", "**/IRX_*.TIF", "**/IRX_*.TIFF",
            "**/irx_*.tif", "**/irx_*.tiff", "**/irx_*.TIF", "**/irx_*.TIFF",
        ]
    else:
        patterns = [
            "IRX_*.tif", "IRX_*.tiff", "IRX_*.TIF", "IRX_*.TIFF",
            "irx_*.tif", "irx_*.tiff", "irx_*.TIF", "irx_*.TIFF",
        ]

    out = []
    for pat in patterns:
        out.extend(glob.glob(os.path.join(root, pat), recursive=RECURSIVE))

    cleaned = []
    for p in sorted(set(out)):
        if find_frame_id(p) is not None:
            cleaned.append(p)
    return cleaned

# --------------------------------------------------
# Thermal + histogram logic
# --------------------------------------------------

def to_celsius_autel(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32) * 0.1 - 273.15

def local_std_texture(temp_c: np.ndarray, finite: np.ndarray, win: int):
    filled = temp_c.copy()
    fill_val = float(np.nanmedian(temp_c[finite])) if np.any(finite) else 0.0
    filled[~finite] = fill_val

    mean = uniform_filter(filled, win)
    mean_sq = uniform_filter(filled**2, win)
    var = np.maximum(mean_sq - mean**2, 0.0)
    return np.sqrt(var)

def smooth_1d(x: np.ndarray, k: int):
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k, dtype=np.float64) / k
    return np.convolve(xp, ker, mode="valid")

def texture_hist(tex_vals: np.ndarray, nbins: int):
    tex_vals = tex_vals[np.isfinite(tex_vals)]
    if tex_vals.size == 0:
        return None, None

    lo, hi = np.percentile(tex_vals, 1), np.percentile(tex_vals, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(tex_vals))
        hi = float(np.max(tex_vals))
        if hi <= lo:
            return None, None

    hist, bin_edges = np.histogram(tex_vals, bins=nbins, range=(lo, hi))
    return hist, bin_edges

def angle_knee_threshold(hist: np.ndarray, bin_edges: np.ndarray, smooth_k: int, angle_deg: float, run: int):
    hist = hist.astype(np.float64)
    s = hist.sum()
    if s > 0:
        hist = hist / s

    hs = smooth_1d(hist, smooth_k)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    peak_idx = int(np.argmax(hs))
    dx = np.diff(centers)
    dy = np.diff(hs)

    dx[dx == 0] = np.nan
    slopes = dy / dx
    angles = np.degrees(np.arctan(slopes))

    thr_idx = None
    for i in range(peak_idx, len(angles) - run):
        window = angles[i:i + run]
        if np.all(np.abs(window) < angle_deg):
            thr_idx = i + 1
            break

    if thr_idx is None:
        for j in range(peak_idx + 1, len(hs) - 1):
            if hs[j - 1] > hs[j] and hs[j] <= hs[j + 1]:
                thr_idx = j
                break

    if thr_idx is None:
        thr_idx = min(peak_idx + 1, len(centers) - 1)

    thr = float(centers[thr_idx])
    return thr, angles, hs, centers

def peak_width_fwhm(hs: np.ndarray, centers: np.ndarray):
    peak_idx = int(np.argmax(hs))
    peak_h = float(hs[peak_idx])
    half = 0.5 * peak_h

    left = peak_idx
    while left > 0 and hs[left] > half:
        left -= 1

    right = peak_idx
    n = len(hs)
    while right < n - 1 and hs[right] > half:
        right += 1

    width = float(centers[right] - centers[left])
    return width, peak_idx, peak_h

def compute_metrics_for_tiff(path: str):
    rec = {
        "file": os.path.basename(path),
        "path": path,
        "frame_id": find_frame_id(path),
        "status": "ok",
        "reason": "",
    }

    try:
        with rasterio.open(path) as src:
            raw = src.read(1)
    except Exception as e:
        rec["status"] = "failed"
        rec["reason"] = f"open_failed: {e}"
        return rec

    if raw.dtype.kind not in ("u", "i", "f"):
        rec["status"] = "failed"
        rec["reason"] = f"unexpected_dtype: {raw.dtype}"
        return rec

    temp_c = to_celsius_autel(raw)
    temp_c[(temp_c < MIN_C) | (temp_c > MAX_C)] = np.nan
    finite = np.isfinite(temp_c)
    if not np.any(finite):
        rec["status"] = "failed"
        rec["reason"] = "no_finite_temps"
        return rec

    tex = local_std_texture(temp_c, finite, TEX_WIN)
    hist, bin_edges = texture_hist(tex[finite], TEX_BINS)
    if hist is None:
        rec["status"] = "failed"
        rec["reason"] = "texture_hist_failed"
        return rec

    thr, angles, hs, centers = angle_knee_threshold(hist, bin_edges, HIST_SMOOTH_K, ANGLE_DEG, ANGLE_RUN)
    width, peak_idx, peak_h = peak_width_fwhm(hs, centers)
    peak_x = float(centers[int(peak_idx)])

    # peak sharpness
    sharpness = float(peak_h / width) if width > 0 else float("nan")

    # peak prominence
    prom = peak_prominences(hs, [peak_idx])[0][0]

    histn = hist.astype(np.float64)
    s = histn.sum()
    if s > 0:
        histn /= s
    area_left = float(histn[centers <= thr].sum())

    rec.update({
        "peak_x": float(peak_x),
        "peak_height": float(peak_h),
        "peak_width_fwhm": float(width),
        "peak_sharpness": float(sharpness),
        "hist_lo": float(bin_edges[0]),
        "hist_hi": float(bin_edges[-1]),
    })  

    return rec

# --------------------------------------------------
# Plotting
# --------------------------------------------------

def add_strip_panel(ax, rows, metric, title):
    good_vals = [float(r[metric]) for r in rows if r["label"] == "good" and metric in r and r["status"] == "ok"]
    bad_vals = [float(r[metric]) for r in rows if r["label"] == "bad" and metric in r and r["status"] == "ok"]

    rng = np.random.default_rng(0)
    yg = 1.0 + rng.uniform(-0.08, 0.08, size=len(good_vals))
    yb = 0.0 + rng.uniform(-0.08, 0.08, size=len(bad_vals))

    if bad_vals:
        ax.scatter(bad_vals, yb, s=35, label="bad", alpha=0.9)
    if good_vals:
        ax.scatter(good_vals, yg, s=35, label="good", alpha=0.9)

    if bad_vals:
        med_bad = float(np.median(bad_vals))
        ax.axvline(med_bad, linestyle="--", linewidth=1.5)
        ax.text(med_bad, -0.28, f"bad med={med_bad:.4g}", rotation=90, va="bottom", ha="right", fontsize=8)
    if good_vals:
        med_good = float(np.median(good_vals))
        ax.axvline(med_good, linestyle=":", linewidth=1.5)
        ax.text(med_good, 1.10, f"good med={med_good:.4g}", rotation=90, va="bottom", ha="left", fontsize=8)

    ax.set_title(title)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["bad", "good"])
    ax.set_xlabel(metric)
    ax.set_ylim(-0.45, 1.35)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper right")

def write_csv(rows, out_csv):
    fixed = [
        "label", "status", "reason", "frame_id", "file", "path",
        "peak_x", "peak_height", "threshold", "peak_width_fwhm",
        "area_left_of_threshold", "hist_lo", "hist_hi"
    ]
    keys = []
    seen = set()
    for k in fixed:
        if k not in seen:
            keys.append(k)
            seen.add(k)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    good_files = list_irx_tiffs(GOOD_DIR)
    bad_files = list_irx_tiffs(BAD_DIR)

    print(f"Good files found: {len(good_files)}")
    print(f"Bad files found:  {len(bad_files)}")

    rows = []

    for p in good_files:
        rec = compute_metrics_for_tiff(p)
        rec["label"] = "good"
        rows.append(rec)

    for p in bad_files:
        rec = compute_metrics_for_tiff(p)
        rec["label"] = "bad"
        rows.append(rec)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(OUTPUT_DIR, f"1d_metrics_{timestamp}.csv")
    out_png = os.path.join(OUTPUT_DIR, f"1d_metrics_{timestamp}.png")

    write_csv(rows, out_csv)

    metrics = [
        ("peak_x", "Peak Position"),
        ("peak_height", "Peak Height"),
        ("peak_width_fwhm", "Peak Width (FWHM)"),
        ("peak_sharpness", "Peak Sharpness"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 3 * len(metrics)), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    ok_rows = [r for r in rows if r["status"] == "ok"]

    for ax, (metric, title) in zip(axes, metrics):
        add_strip_panel(ax, ok_rows, metric, title)

    n_good_ok = sum(1 for r in rows if r["label"] == "good" and r["status"] == "ok")
    n_bad_ok = sum(1 for r in rows if r["label"] == "bad" and r["status"] == "ok")
    n_fail = sum(1 for r in rows if r["status"] != "ok")

    fig.suptitle(
        f"1D Histogram Metric Comparison\n"
        f"good={n_good_ok}, bad={n_bad_ok}, failed={n_fail}",
        fontsize=14
    )

    fig.savefig(out_png, dpi=180)
    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote PNG: {out_png}")

    plt.show()

if __name__ == "__main__":
    main()
