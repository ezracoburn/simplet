#!/usr/bin/env python3
import os
import re
import glob
import json
import math
import subprocess
from datetime import datetime

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, label

# -----------------------------
# USER SETTINGS (EDIT THESE)
# -----------------------------
FLIGHT_ROOT = '/Volumes/EXTERNAL HD/Thermal Flights/1 July 23/Hanga Roa - Rano Kau'

# Put EVERYTHING (metrics + scatter PNGs) directly in this folder (no extra subfolders)
OUTPUT_DIR = "/Users/ezracoburn/Documents/Simple/output/flight_hist_metrics_scatter"

# RGB files are always MAX_####.JPG (per your note)
RGB_PATTERN = "MAX_{id}.JPG"

# Thermal JPG pattern (adjust if yours differs)
THERMAL_JPG_PATTERN = "IRX_{id}.JPG"

# Click behavior
OPEN_THERMAL_JPG_ON_CLICK = True
OPEN_RGB_ON_CLICK = True
SHOW_HIST_ON_CLICK = True   # shows histogram + angle plots (matplotlib) on click

# Histogram/texture params (keep consistent with your pipeline defaults)
TEX_BINS = 256
TEX_WIN = 9
HIST_SMOOTH_K = 9
ANGLE_DEG = 1.0
ANGLE_RUN = 10
CONNECTIVITY_8 = True
MIN_SMOOTH_FRAC = 0.10
MIN_C = -50.0
MAX_C = 200.0

# -----------------------------
# Discovery
# -----------------------------
IRX_RE = re.compile(r"IRX_(\d{4})", re.IGNORECASE)

def find_frame_id(path: str):
    m = IRX_RE.search(os.path.basename(path))
    return m.group(1) if m else None

def find_media_dirs(flight_root: str):
    media_dirs = []
    for d in glob.glob(os.path.join(flight_root, "**", "*MEDIA"), recursive=True):
        if os.path.isdir(d) and re.search(r"[\\/]\d{3}MEDIA$", d):
            media_dirs.append(d)
    return sorted(set(media_dirs))

def list_camera_tiffs(flight_root: str):
    media_dirs = find_media_dirs(flight_root)
    search_roots = media_dirs if media_dirs else [flight_root]

    patterns = [
        "IRX_*.tif", "IRX_*.tiff", "IRX_*.TIF", "IRX_*.TIFF",
        "irx_*.tif", "irx_*.tiff", "irx_*.TIF", "irx_*.TIFF",
    ]

    candidates = []
    for root in search_roots:
        for pat in patterns:
            candidates.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))

    out = []
    for p in sorted(set(candidates)):
        fid = find_frame_id(p)
        if fid is None:
            continue
        out.append(p)

    return out, media_dirs

def _first_glob(root: str, pattern: str):
    hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    hits = sorted(set(hits))
    return hits[0] if hits else None

def find_rgb_for_frame_id(flight_root: str, frame_id: str):
    if not frame_id:
        return None
    # exact pattern first (fast)
    p = _first_glob(flight_root, RGB_PATTERN.format(id=frame_id))
    if p:
        return p
    # case variants
    p = _first_glob(flight_root, RGB_PATTERN.format(id=frame_id).lower())
    return p

def find_thermal_jpg_for_frame_id(flight_root: str, frame_id: str):
    if not frame_id:
        return None
    p = _first_glob(flight_root, THERMAL_JPG_PATTERN.format(id=frame_id))
    if p:
        return p
    p = _first_glob(flight_root, THERMAL_JPG_PATTERN.format(id=frame_id).lower())
    return p

def open_file(path: str):
    if not path or not os.path.exists(path):
        return
    subprocess.run(["open", path], check=False)

# -----------------------------
# Thermal + texture + histogram
# -----------------------------
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

def smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
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

def show_hist_angle_for_tiff(tiff_path: str, title_base: str = ""):
    """Display histogram+threshold and slope-angle plot (same logic as the pipeline debug)."""
    with rasterio.open(tiff_path) as src:
        raw = src.read(1)

    temp_c = to_celsius_autel(raw)
    temp_c[(temp_c < MIN_C) | (temp_c > MAX_C)] = np.nan
    finite = np.isfinite(temp_c)
    if not np.any(finite):
        return

    tex = local_std_texture(temp_c, finite, TEX_WIN)
    hist, bin_edges = texture_hist(tex[finite], TEX_BINS)
    if hist is None:
        return

    thr, angles, hs, centers = angle_knee_threshold(hist, bin_edges, HIST_SMOOTH_K, ANGLE_DEG, ANGLE_RUN)

    histn = hist.astype(np.float64)
    s = histn.sum()
    if s > 0:
        histn /= s

    # 1) histogram
    plt.figure(figsize=(7, 4))
    plt.plot(centers, histn, alpha=0.35, label="hist (norm)")
    plt.plot(centers, hs, label=f"smoothed (k={HIST_SMOOTH_K})")
    plt.axvline(thr, linewidth=2, label="thr")
    plt.title(f"{title_base}\nthr={thr:.6f}".strip())
    plt.xlabel("Texture (local std of °C)")
    plt.ylabel("Probability mass")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    # 2) slope-angle
    plt.figure(figsize=(7, 3))
    plt.plot(centers[1:], angles, label="slope angle (deg)")
    plt.axhline(ANGLE_DEG, linestyle="--", linewidth=1, label="±ANGLE_DEG")
    plt.axhline(-ANGLE_DEG, linestyle="--", linewidth=1)
    plt.axvline(thr, linewidth=2, label="thr")
    plt.title(f"{title_base}\nANGLE_DEG={ANGLE_DEG}, RUN={ANGLE_RUN}".strip())
    plt.xlabel("Texture")
    plt.ylabel("Angle (deg)")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    plt.pause(0.001)

# -----------------------------
# Segmentation S2 (largest component)
# -----------------------------
def label_mask(mask: np.ndarray, connectivity_8: bool):
    if connectivity_8:
        structure = np.ones((3, 3), dtype=np.int32)
    else:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.int32)
    return label(mask, structure=structure)

def keep_largest_component(mask: np.ndarray, connectivity_8: bool):
    lbl, n = label_mask(mask, connectivity_8)
    if n == 0:
        return mask & False
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    keep_id = int(np.argmax(counts))
    return lbl == keep_id

# -----------------------------
# Scatter (build figure; single plt.show at end)
# -----------------------------
def build_scatter_figure(points, x_key, y_key, out_png, title=None):
    xs = np.array([p[x_key] for p in points], dtype=np.float64)
    ys = np.array([p[y_key] for p in points], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(xs, ys, s=18)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title or f"{x_key} vs {y_key}")

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        xpix, ypix = ax.transData.transform(np.column_stack([xs, ys])).T
        dx = xpix - event.x
        dy = ypix - event.y
        dist2 = dx * dx + dy * dy
        i = int(dist2.argmin())

        if dist2[i] > (18 * 18):
            return

        p = points[i]
        fid = p.get("frame_id")
        base = p.get("file")
        status = p.get("status")
        step = p.get("step") or ""
        reason = p.get("reason") or ""
        tiff_path = p.get("path")

        label = f"IRX_{fid} | {status}"
        if step or reason:
            label += f" | {step}:{reason}"
        label += f" | {base} | {x_key}={p.get(x_key)} | {y_key}={p.get(y_key)}"
        print(label)

        if OPEN_THERMAL_JPG_ON_CLICK and fid:
            therm = find_thermal_jpg_for_frame_id(FLIGHT_ROOT, fid)
            if therm:
                open_file(therm)

        if OPEN_RGB_ON_CLICK and fid:
            rgb = find_rgb_for_frame_id(FLIGHT_ROOT, fid)
            if rgb:
                open_file(rgb)

        if SHOW_HIST_ON_CLICK and tiff_path:
            show_hist_angle_for_tiff(tiff_path, title_base=f"IRX_{fid}")

    fig.canvas.mpl_connect("button_press_event", on_click)

    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=160)
    return fig

# -----------------------------
# Main metric extraction
# -----------------------------
def process_tiff(path: str):
    base = os.path.basename(path)
    frame_id = find_frame_id(path)

    rec = {
        "file": base,
        "path": path,
        "frame_id": frame_id,
        "status": "ok",
        "step": None,
        "reason": None,
    }

    try:
        with rasterio.open(path) as src:
            raw = src.read(1)
    except Exception as e:
        rec.update(status="skipped", step="read_tiff", reason="open_failed")
        rec["error"] = str(e)
        return rec

    if raw.dtype.kind not in ("u", "i", "f"):
        rec.update(status="skipped", step="read_tiff", reason="unexpected_dtype")
        rec["dtype"] = str(raw.dtype)
        return rec

    temp_c = to_celsius_autel(raw)
    temp_c[(temp_c < MIN_C) | (temp_c > MAX_C)] = np.nan
    finite = np.isfinite(temp_c)
    if not np.any(finite):
        rec.update(status="skipped", step="temp_sanitize", reason="no_finite_temps")
        return rec

    tex = local_std_texture(temp_c, finite, TEX_WIN)
    hist, bin_edges = texture_hist(tex[finite], TEX_BINS)
    if hist is None:
        rec.update(status="skipped", step="texture_hist", reason="texture_hist_failed")
        return rec

    thr, angles, hs, centers = angle_knee_threshold(hist, bin_edges, HIST_SMOOTH_K, ANGLE_DEG, ANGLE_RUN)

    width, peak_idx, peak_h = peak_width_fwhm(hs, centers)
    peak_x = float(centers[int(peak_idx)])

    histn = hist.astype(np.float64)
    s = histn.sum()
    if s > 0:
        histn /= s
    area_left = float(histn[centers <= thr].sum())

    rec.update({
        "threshold": float(thr),
        "peak_x": float(peak_x),
        "peak_height": float(peak_h),
        "peak_width_fwhm": float(width),
        "area_left_of_threshold": float(area_left),
        "hist_lo": float(bin_edges[0]),
        "hist_hi": float(bin_edges[-1]),
    })

    S = finite & (tex <= thr)
    S2 = keep_largest_component(S, CONNECTIVITY_8)

    frac_finite = float(np.count_nonzero(finite))
    rec["s_frac"] = float(np.count_nonzero(S)) / frac_finite
    rec["s2_frac"] = float(np.count_nonzero(S2)) / frac_finite

    if rec["s2_frac"] < MIN_SMOOTH_FRAC:
        rec["status"] = "baseline_skipped"
        rec["step"] = "baseline"
        rec["reason"] = "s2_frac_below_min"

    return rec

def main():
    if not os.path.isdir(FLIGHT_ROOT):
        raise SystemExit(f"FLIGHT_ROOT does not exist: {FLIGHT_ROOT}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tiffs, media_dirs = list_camera_tiffs(FLIGHT_ROOT)
    if not tiffs:
        print(f"No camera TIFFs found under: {FLIGHT_ROOT}")
        if media_dirs:
            print(f"Found {len(media_dirs)} ###MEDIA dirs (example):")
            for d in media_dirs[:5]:
                print(" ", d)
        raise SystemExit(1)

    print(f"Found {len(tiffs)} TIFFs. Processing…")
    recs = []
    for i, p in enumerate(tiffs, 1):
        recs.append(process_tiff(p))
        if i % 50 == 0:
            print(f"  {i}/{len(tiffs)}")

    now = datetime.now().isoformat(timespec="seconds")
    out_json = os.path.join(OUTPUT_DIR, "flight_metrics.json")
    out_csv = os.path.join(OUTPUT_DIR, "flight_metrics.csv")

    payload = {
        "generated_at": now,
        "flight_root": FLIGHT_ROOT,
        "params": {
            "tex_bins": TEX_BINS,
            "tex_win": TEX_WIN,
            "hist_smooth_k": HIST_SMOOTH_K,
            "angle_deg": ANGLE_DEG,
            "angle_run": ANGLE_RUN,
            "connectivity_8": CONNECTIVITY_8,
            "min_smooth_frac": MIN_SMOOTH_FRAC,
            "min_c": MIN_C,
            "max_c": MAX_C,
        },
        "records": recs
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    keys = sorted({k for r in recs for k in r.keys()})
    with open(out_csv, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in recs:
            row = []
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, str):
                    v = v.replace('"', '""')
                    if "," in v or "\n" in v:
                        v = f'"{v}"'
                row.append(str(v))
            f.write(",".join(row) + "\n")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")

    def valid(p, k):
        v = p.get(k)
        if v is None:
            return False
        try:
            v = float(v)
        except Exception:
            return False
        return math.isfinite(v)

    points = [r for r in recs if valid(r, "threshold")]

    scatters = [
       # ("peak_width_fwhm", "threshold"),
       # ("area_left_of_threshold", "threshold"),
        ("peak_height", "peak_width_fwhm"),
      #  ("s2_frac", "peak_width_fwhm"),
    ]

    figs = []
    for xk, yk in scatters:
        pts = [p for p in points if valid(p, xk) and valid(p, yk)]
        if not pts:
            print(f"Skip plot {xk} vs {yk}: no valid points.")
            continue

        out_png = os.path.join(OUTPUT_DIR, f"scatter_{xk}_vs_{yk}.png")
        title = f"{os.path.basename(FLIGHT_ROOT)}\n{xk} vs {yk} (click: open thermal jpg + rgb + hist)"
        figs.append(build_scatter_figure(pts, xk, yk, out_png, title=title))

    # IMPORTANT:
    # - interactive clicking works only in these live matplotlib windows.
    # - PNG files saved to disk will NOT be clickable.
    plt.show()

if __name__ == "__main__":
    main()
