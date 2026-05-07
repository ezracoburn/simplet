import os
import re
import glob
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import exifread

from pyproj import CRS, Transformer
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union, transform as shapely_transform

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import shapes
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, label, distance_transform_edt, gaussian_filter, median_filter, gaussian_filter1d
from scipy.signal import find_peaks
from collections import Counter
import json
from datetime import datetime

OUTPUT_ROOT = "/Users/ezracoburn/Documents/Simple/output/5-5_bursts_bc/Tongariki Clipped - lowerpk"

BYFRAME_DIR = os.path.join(OUTPUT_ROOT, "byframe")
PASSES_DIR = os.path.join(BYFRAME_DIR, "passes")
CUTS_DIR = os.path.join(BYFRAME_DIR, "cuts")
LAYER_DIR = os.path.join(OUTPUT_ROOT, "layers")
BIAS_FIELD_DIR = os.path.join(OUTPUT_ROOT, "bias_field")
RASTER_LAYER_DIR = os.path.join(LAYER_DIR, "rasters")
POLYGON_LAYER_DIR = os.path.join(LAYER_DIR, "polygons")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(BYFRAME_DIR, exist_ok=True)
os.makedirs(PASSES_DIR, exist_ok=True)
os.makedirs(CUTS_DIR, exist_ok=True)
os.makedirs(LAYER_DIR, exist_ok=True)
os.makedirs(BIAS_FIELD_DIR, exist_ok=True)
os.makedirs(RASTER_LAYER_DIR, exist_ok=True)
os.makedirs(POLYGON_LAYER_DIR, exist_ok=True)

# georeferencing constants
THERMAL_HFOV_DEG = 33.0
THERMAL_VFOV_DEG = 26.0
THERMAL_WIDTH_PX = 640
THERMAL_HEIGHT_PX = 512
HIGH_ALTITUDE_WARNING_M = 450.0

BURST_SIZE = 1000
NUM_BURSTS = 1

# total selected frames will be BURST_SIZE * NUM_BURSTS (subject to availability)
NUM_IMAGES = BURST_SIZE * NUM_BURSTS

# bias field constants
APPLY_BIAS_CORRECTION = True
BIAS_MIN_COUNT = 10
BIAS_SMOOTH_SIGMA = 30
USE_DIRECTIONAL_BIAS = True

YAW_HIST_BIN_DEG = 5
YAW_HIST_SMOOTH_SIGMA_BINS = 1
YAW_PEAK_MIN_DISTANCE_DEG = 35
YAW_PEAK_SUPPORT_WINDOW_DEG = 15
YAW_PEAK_MIN_RAW_COUNT = 25  # should be > bias_min_count, so all peaks can build a bias field

# mask building constants
MIN_SMOOTH_FRAC = 0.10
S2_OVER_S_MIN = 0.8
BASELINE_PERCENTILE = 90
BASELINE_BAND_DELTA_C = 0.1
CONTOUR_STEP_C = 0.25

# cumulative cold-mask aggregation constants
BUILD_CUMULATIVE_COLD_MASKS = True
COLD_MASK_MIN_DELTA_C = 0.25
COLD_MASK_MAX_DELTA_C = 1.0

# aggregation constants
BUILD_AGGREGATE_MASK_LAYERS = True
AGG_GRID_RES_M = 0.5                    # should be > max GSD (~0.3 m for our flights)
MIN_SUPPORT_FRACTION = 0.25
MIN_S2_SUPPORT_COUNT = 2

# polygon output constants
BUILD_AGGREGATE_POLYGONS = True
MIN_POLYGON_AREA_M2 = 0.0

# texture histogram building constants
TEX_BINS = 256
TEX_WIN = 9

# Angle-knee params 
    # Main smoothing (HIST_SMOOTH_K) is used for thresholding and stable peak geometry.
    # A lighter smoothing (PEAK_HEIGHT_K) is used only to compute rawer_peak_height,
    # which is intended to be a more sensitive QC metric for tall/narrow ocean peaks.
    # kernel must be odd
HIST_SMOOTH_K = 7 
ANGLE_DEG = 0.7
ANGLE_RUN = 6
PEAK_HEIGHT_K = 3


# histogram metric filters
# primary filter
PEAK_HEIGHT_MIN = 0.065 
# liberal sanity check filters
TEXTURE_THR_MAX = 0.5
PEAK_X_MAX = 0.15
PEAK_WIDTH_MAX = 0.2
PEAK_SHARPNESS_MIN = 0.1

SAVE_CUT_DEBUG = True
CUT_DEBUG_DPI = 120

# S2 mode: keep only pixels within X pixels of the largest component
S2_MODE = "largest"      # "largest" | "edge" | "within_x"
CONNECTIVITY_8 = True

DILATE_PIXELS = 0        # X pixels: keep S pixels with distance-to-largest <= X

# Optional debug output
SAVE_DIST_HEATMAP = False
DIST_HEATMAP_CLIP = 30   # clip distances for visualization (pixels)
                         #CANNOT SET TO 0 (runtime)

FILTERS_ENABLED = {
    "thr_max": True,
    "peak_x_max": True,
    "peak_height_min": True,
    "peak_width_max": True,
    "peak_sharpness_min": False,
    "s2_min_frac_baseline_skip": True,
    "s2_over_s_min": True,
}


# -------------------------
# REPORTING HELPERS
# -------------------------


def get_run_constants():
    return {
        name: value
        for name, value in globals().items()
        if name.isupper()
        and isinstance(value, (str, int, float, bool, type(None), list, tuple, dict))
    }


def _new_report():
    return {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "constants": get_run_constants(),
        "filters_enabled": dict(FILTERS_ENABLED),
        "num_selected": 0,
        "num_processed": 0,          # process_one reached the end (produced overlays)
        "num_skipped": 0,            # process_one returned early
        "skipped_by_reason": Counter(),
        "num_baseline_skipped": 0,   # processed but baseline not computed
        "baseline_skipped_by_reason": Counter(),
        "frames": [],                # per-frame records (extensible)
        "cuts_by_step": {},          # step -> reason -> [files]
    }


def _report_add_cut(report, step: str, reason: str, base: str):
    cuts = report["cuts_by_step"].setdefault(step, {})
    cuts.setdefault(reason, []).append(base)


def _report_add_skip(report, frame_id, base, reason, details=None, step="unknown"):
    report["num_skipped"] += 1
    report["skipped_by_reason"][reason] += 1
    _report_add_cut(report, step, reason, base)
    report["frames"].append({
        "frame_id": frame_id,
        "file": base,
        "status": "skipped",
        "step": step,
        "reason": reason,
        "details": details or {}
    })


def _report_add_processed(report, frame_id, base, details=None):
    report["num_processed"] += 1
    report["frames"].append({
        "frame_id": frame_id,
        "file": base,
        "status": "processed",
        "details": details or {}
    })


def _report_add_baseline_skip(report, frame_id, base, reason, details=None, step="baseline"):
    report["num_baseline_skipped"] += 1
    report["baseline_skipped_by_reason"][reason] += 1
    _report_add_cut(report, step, reason, base)
    report["frames"].append({
        "frame_id": frame_id,
        "file": base,
        "status": "baseline_skipped",
        "step": step,
        "reason": reason,
        "details": details or {}
    })


def write_report(report, output_root):
    report_out = dict(report)
    report_out["skipped_by_reason"] = dict(report["skipped_by_reason"])
    report_out["baseline_skipped_by_reason"] = dict(report["baseline_skipped_by_reason"])
    report_out["cuts_by_step"] = report.get("cuts_by_step", {})
    report_out["finished_at"] = datetime.now().isoformat(timespec="seconds")

    path_json = os.path.join(OUTPUT_ROOT, "run_report.json")
    with open(path_json, "w") as f:
        json.dump(report_out, f, indent=2)


def write_text_report(report, output_root):
    path_txt = os.path.join(OUTPUT_ROOT, "run_report.txt")
    with open(path_txt, "w") as f:
        f.write("=== RUN REPORT ===\n")
        f.write(f"started_at: {report['started_at']}\n")
        f.write(f"finished_at: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write(f"num_selected: {report['num_selected']}\n")
        f.write(f"num_processed: {report['num_processed']}\n")
        f.write(f"num_skipped: {report['num_skipped']}\n")
        f.write(f"num_baseline_skipped: {report['num_baseline_skipped']}\n\n")

        f.write("=== CUTS BY STEP ===\n")
        cuts = report.get("cuts_by_step", {})
        if not cuts:
            f.write("(none)\n")
            return

        for step in sorted(cuts.keys()):
            f.write(f"\n[{step}]\n")
            for reason in sorted(cuts[step].keys()):
                files = cuts[step][reason]
                f.write(f"  - {reason}: {len(files)}\n")
                for name in files:
                    f.write(f"      {name}\n")


# -------------------------
# FILE DISCOVERY / SELECTION HELPERS
# -------------------------


def find_frame_id(path: str):
    m = re.search(r"IRX_(\d{4})", os.path.basename(path))
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
        "IRX_*.tif", "IRX_*.tiff",
        "IRX_*.TIF", "IRX_*.TIFF",
        "irx_*.tif", "irx_*.tiff",
        "irx_*.TIF", "irx_*.TIFF",
    ]

    candidates = []
    for root in search_roots:
        for pat in patterns:
            candidates.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))

    out = []
    for p in sorted(set(candidates)):
        name = os.path.basename(p)
        if not re.fullmatch(r"IRX_\d{4}\.(tif|tiff|TIF|TIFF)", name):
            continue
        fid = find_frame_id(p)
        if fid is None:
            continue
        base = name.lower()
        if "rpeg" in base or "rgb" in base or "mosaic" in base or "orth" in base:
            continue
        out.append(p)

    return out, media_dirs


def pick_bursts(items, burst_size: int, num_bursts: int):
    n = len(items)
    if n == 0 or burst_size <= 0 or num_bursts <= 0:
        return []

    if n <= burst_size:
        return items

    max_start = n - burst_size
    if num_bursts == 1:
        starts = [0]
    else:
        starts = np.linspace(0, max_start, num_bursts, dtype=int).tolist()

    selected = []
    used = set()
    for s in starts:
        for i in range(s, min(s + burst_size, n)):
            if i not in used:
                selected.append(items[i])
                used.add(i)

    return selected


# -------------------------
# TEMPERATURE / DISPLAY HELPERS
# -------------------------


def to_celsius_autel(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32) * 0.1 - 273.15


def normalize_for_display(img: np.ndarray):
    finite = np.isfinite(img)
    v = img[finite]
    lo = np.percentile(v, 2)
    hi = np.percentile(v, 98)
    if hi <= lo:
        hi = lo + 1.0
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def overlay_and_save(background, mask, title, out_path, alpha=0.45):
    plt.figure(figsize=(8, 6))
    plt.imshow(background, cmap="gray", vmin=0.0, vmax=1.0)

    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = mask.astype(np.float32) * alpha
    plt.imshow(overlay)

    plt.title(title)
    plt.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# -------------------------
# TEXTURE / HISTOGRAM HELPERS
# -------------------------


def local_std_texture(temp_c: np.ndarray, finite: np.ndarray, win: int = 9):
    filled = temp_c.copy()
    fill_val = float(np.nanmedian(temp_c[finite])) if np.any(finite) else 0.0
    filled[~finite] = fill_val

    mean = uniform_filter(filled, win)
    mean_sq = uniform_filter(filled**2, win)
    var = np.maximum(mean_sq - mean**2, 0)
    return np.sqrt(var)


def smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    if k % 2 == 0:
        raise ValueError(f"smooth_1d requires odd k, got {k}")
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k, dtype=np.float64) / k
    return np.convolve(xp, ker, mode="valid")


def angle_knee_threshold(hist: np.ndarray, bin_edges: np.ndarray,
                         smooth_k: int = HIST_SMOOTH_K,
                         angle_deg: float = ANGLE_DEG,
                         run: int = ANGLE_RUN):
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

    min_idx = peak_idx + int(np.argmin(angles[peak_idx:]))

    thr_idx = None
    for i in range(min_idx, len(angles) - run):
        window = angles[i:i + run]
        if np.all(window > -angle_deg):
            thr_idx = i + 1
            break

    if thr_idx is None:
        return None, angles, hs, centers

    thr = float(centers[thr_idx])
    return thr, angles, hs, centers


def texture_hist(tex_vals: np.ndarray, nbins: int = TEX_BINS):
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


def compute_hist_metrics(hist: np.ndarray, hs: np.ndarray, centers: np.ndarray, thr: float):
    # Main metrics from the main smoothed histogram (HIST_SMOOTH_K)
    width, peak_idx, peak_h = peak_width_fwhm(hs, centers)
    peak_x = float(centers[peak_idx])
    sharpness = float(peak_h / width) if width > 0 else float("nan")

    # Normalize raw histogram, then make a lighter-smoothed version
    histn = hist.astype(np.float64)
    s = histn.sum()
    if s > 0:
        histn /= s

    hs_height = smooth_1d(histn, PEAK_HEIGHT_K)
    rawer_peak_h = float(np.max(hs_height))

    return {
        "threshold": float(thr),
        "peak_x": peak_x,
        "peak_height": float(peak_h),
        "rawer_peak_height": float(rawer_peak_h),
        "peak_width_fwhm": float(width),
        "peak_sharpness": float(sharpness),
    }


# -------------------------
# MASK CONSTRUCTION HELPERS
# -------------------------


def label_mask(mask: np.ndarray, connectivity_8: bool = True):
    if connectivity_8:
        structure = np.ones((3, 3), dtype=np.int32)
    else:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.int32)
    return label(mask, structure=structure)


def keep_largest_component(mask: np.ndarray, connectivity_8: bool = True):
    lbl, n = label_mask(mask, connectivity_8)
    if n == 0:
        return mask & False
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    keep_id = int(np.argmax(counts))
    return lbl == keep_id


def keep_edge_components(mask: np.ndarray, connectivity_8: bool = True):
    lbl, n = label_mask(mask, connectivity_8)
    if n == 0:
        return mask & False
    edge_ids = np.unique(np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]]))
    edge_ids = edge_ids[edge_ids != 0]
    if edge_ids.size == 0:
        return keep_largest_component(mask, connectivity_8)
    return np.isin(lbl, edge_ids)


def keep_within_x_of_largest(mask: np.ndarray, x_pixels: int, connectivity_8: bool = True):
    L = keep_largest_component(mask, connectivity_8)
    if not np.any(L):
        return mask & False, None, L

    dist = distance_transform_edt(~L)
    keep = mask & (dist <= float(x_pixels))
    return keep, dist, L


def build_s2(mask_s: np.ndarray):
    if S2_MODE == "edge":
        return keep_edge_components(mask_s, CONNECTIVITY_8), None, None
    if S2_MODE == "largest":
        L = keep_largest_component(mask_s, CONNECTIVITY_8)
        return L, None, L
    return keep_within_x_of_largest(mask_s, DILATE_PIXELS, CONNECTIVITY_8)


def cold_threshold_values():
    n_steps = int(np.floor(COLD_MASK_MAX_DELTA_C / COLD_MASK_MIN_DELTA_C))
    return [
        round(i * COLD_MASK_MIN_DELTA_C, 2)
        for i in range(1, n_steps + 1)
    ]


def threshold_label(delta_c: float):
    return f"{delta_c:.2f}".replace(".", "p")


def build_cumulative_cold_masks(diff_c: np.ndarray, s2_mask: np.ndarray):
    valid = s2_mask & np.isfinite(diff_c)

    masks = {}
    for delta_c in cold_threshold_values():
        masks[delta_c] = valid & (diff_c <= -float(delta_c))

    return valid, masks


# -------------------------
# CUT / PASS DEBUG OUTPUT HELPERS
# -------------------------


def _cut_dir(step: str, reason: str):
    d = os.path.join(CUTS_DIR, step, reason)
    os.makedirs(d, exist_ok=True)
    return d


def _pass_path(filename: str) -> str:
    return os.path.join(PASSES_DIR, filename)


def save_cut_debug(frame_id: str, base: str, temp_c: np.ndarray,
                   hist, bin_edges, thr, angles, hs, centers,
                   step: str, reason: str):

    if not SAVE_CUT_DEBUG or not frame_id:
        return

    outdir = _cut_dir(step, reason)

    # ---- thermal preview from TIFF ----
    disp = normalize_for_display(temp_c)

    plt.figure(figsize=(6, 5))
    plt.imshow(disp, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title(base)
    plt.axis("off")
    plt.savefig(
        os.path.join(outdir, f"IRX_{frame_id}_thermal.png"),
        dpi=CUT_DEBUG_DPI,
        bbox_inches="tight"
    )
    plt.close()

    # ---- histogram ----
    histn = hist.astype(np.float64)
    s = histn.sum()
    if s > 0:
        histn /= s
    
    hs_height = smooth_1d(histn, PEAK_HEIGHT_K)

    plt.figure(figsize=(7, 4))
    plt.plot(centers, histn, alpha=0.35, label="hist (norm)")
    plt.plot(centers, hs, label=f"smoothed (k={HIST_SMOOTH_K})")
    plt.plot(centers, hs_height, "g--", alpha=0.7, label=f"smoothed (k={PEAK_HEIGHT_K})")
    plt.axvline(thr, linewidth=2, label="thr")
    plt.title(f"{base}\nthr={thr:.6f}")
    plt.xlabel("Texture (local std of °C)")
    plt.ylabel("Probability mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, f"IRX_{frame_id}_texture_hist.png"),
        dpi=CUT_DEBUG_DPI
    )
    plt.close()


def save_hist_angle_debug(frame_id: str, base: str, hist, bin_edges, thr, angles, hs, centers):
    histn = hist.astype(np.float64)
    if histn.sum() > 0:
        histn /= histn.sum()

    hs_height = smooth_1d(histn, PEAK_HEIGHT_K)

    plt.figure(figsize=(8, 4))
    plt.plot(centers, histn, alpha=0.35, label="hist (norm)")
    plt.plot(centers, hs, label=f"smoothed (k={HIST_SMOOTH_K})")
    plt.plot(centers, hs_height, "g--", alpha=0.7, label=f"smoothed (k={PEAK_HEIGHT_K})")
    plt.axvline(thr, linewidth=2, label="angle-knee thr")
    plt.title(f"{base}\nAngle-knee thr={thr:.6f}  angle<{ANGLE_DEG}° run={ANGLE_RUN}")
    plt.xlabel("Texture (local std of °C)")
    plt.ylabel("Probability mass")
    plt.legend()
    plt.tight_layout()
    out_hist = _pass_path(f"IRX_{frame_id}_texture_hist_angleknee.png")
    plt.savefig(out_hist, dpi=160)
    plt.close()

    x = centers[1:]
    plt.figure(figsize=(8, 4))
    plt.plot(x, angles, label="slope angle (deg)")
    plt.axhline(ANGLE_DEG, linestyle="--", linewidth=1, label=f"+{ANGLE_DEG}°")
    plt.axhline(-ANGLE_DEG, linestyle="--", linewidth=1, label=f"-{ANGLE_DEG}°")
    plt.axvline(thr, linewidth=2, label="thr")
    plt.title(f"{base}\nAngle of smoothed hist slope (arctan(dy/dx))")
    plt.xlabel("Texture (local std of °C)")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.tight_layout()
    out_ang = _pass_path(f"IRX_{frame_id}_texture_angle_angleknee.png")
    plt.savefig(out_ang, dpi=160)
    plt.close()


def add_cut_kept_panel(ax, rows, metric, title, cutoff=None, cutoff_label=None):
    kept_vals = [float(r["details"][metric]) for r in rows
                 if r["status"] in ("processed", "baseline_skipped")
                 and "details" in r and metric in r["details"]]

    cut_vals = [float(r["details"][metric]) for r in rows
                if r["status"] == "skipped"
                and "details" in r and metric in r["details"]]

    rng = np.random.default_rng(0)
    y_kept = 1.0 + rng.uniform(-0.08, 0.08, size=len(kept_vals))
    y_cut = 0.0 + rng.uniform(-0.08, 0.08, size=len(cut_vals))

    if cut_vals:
        ax.scatter(cut_vals, y_cut, s=35, alpha=0.9, label="cut")
    if kept_vals:
        ax.scatter(kept_vals, y_kept, s=35, alpha=0.9, label="kept")

    if cutoff is not None:
        ax.axvline(float(cutoff), linestyle="--", linewidth=1.5)
        if cutoff_label:
            ax.text(float(cutoff), 1.18, cutoff_label, rotation=90,
                    va="bottom", ha="left", fontsize=8)

    ax.set_title(title)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["cut", "kept"])
    ax.set_xlabel(metric)
    ax.set_ylim(-0.45, 1.35)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper right")


def save_1d_metrics_plot(report, output_root):
    rows = report.get("frames", [])

    metrics = [
        ("peak_x", "Peak Position", PEAK_X_MAX, "max"),
        ("rawer_peak_height", f"Rawer Peak Height (k={PEAK_HEIGHT_K})", PEAK_HEIGHT_MIN, "min"),
        ("s2_over_s", "S2 / S", S2_OVER_S_MIN, "min"),
        ("threshold", "Threshold", TEXTURE_THR_MAX, "max"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 3 * len(metrics)), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, title, cutoff, kind) in zip(axes, metrics):
        label = f"{kind}={cutoff:.4g}" if cutoff is not None else None
        add_cut_kept_panel(ax, rows, metric, title, cutoff=cutoff, cutoff_label=label)

    fig.suptitle(
        f"1D Histogram Metric Comparison (cut vs kept)\n"
        f"processed={report['num_processed']}, baseline_skipped={report['num_baseline_skipped']}, skipped={report['num_skipped']}",
        fontsize=14
    )

    out_png = os.path.join(OUTPUT_ROOT, "1d_metrics_cut_vs_kept.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_dist_heatmap(frame_id: str, base: str, disp_gray: np.ndarray, dist: np.ndarray, L: np.ndarray):
    if dist is None or L is None:
        return

    d = np.clip(dist, 0, DIST_HEATMAP_CLIP).astype(np.float32)
    d_norm = d / float(DIST_HEATMAP_CLIP)

    plt.figure(figsize=(8, 6))
    plt.imshow(d_norm, cmap="magma", vmin=0.0, vmax=1.0)
    plt.title(f"{base}\nDistance-to-largest heatmap (0..{DIST_HEATMAP_CLIP}px clipped)")
    plt.axis("off")
    out1 = _pass_path(f"IRX_{frame_id}_dist_to_largest_heatmap.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(disp_gray, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imshow(d_norm, cmap="magma", alpha=0.55, vmin=0.0, vmax=1.0)

    outline = np.zeros((*L.shape, 4), dtype=np.float32)
    outline[..., 0] = 0.0
    outline[..., 1] = 1.0
    outline[..., 2] = 1.0
    outline[..., 3] = L.astype(np.float32) * 0.25
    plt.imshow(outline)

    plt.title(f"{base}\nDist-to-largest over thermal (magma), cyan=largest component")
    plt.axis("off")
    out2 = _pass_path(f"IRX_{frame_id}_dist_to_largest_on_gray.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()


# -------------------------
# GEOREFERENCING HELPERS
# -------------------------


def find_matching_jpg(tiff_path: str):
    root, _ = os.path.splitext(tiff_path)
    candidates = [
        root + ".jpg",
        root + ".JPG",
        root + ".jpeg",
        root + ".JPEG",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


@dataclass(frozen=True)
class FrameMeta:
    lat: float
    lon: float
    yaw_deg: float
    alt_agl_m: float
    alt_msl_m: Optional[float]


def _ratio_to_float(r) -> float:
    return float(r.num) / float(r.den)


def _dms_to_deg(dms, ref) -> float:
    deg = _ratio_to_float(dms.values[0])
    minutes = _ratio_to_float(dms.values[1])
    seconds = _ratio_to_float(dms.values[2])
    val = deg + minutes / 60.0 + seconds / 3600.0
    if str(ref.values).strip() in ("S", "W"):
        val = -val
    return val


_XMP_BLOCK_RE = re.compile(rb"<x:xmpmeta.*?</x:xmpmeta>", re.DOTALL)
_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _extract_xmp(jpg_path: str) -> bytes:
    with open(jpg_path, "rb") as f:
        data = f.read()
    m = _XMP_BLOCK_RE.search(data)
    return m.group(0) if m else b""


def _xmp_get_float(xmp: bytes, key: str) -> Optional[float]:
    pat = re.compile(rb"%s\s*=\s*\"([^\"]+)\"" % key.encode())
    m = pat.search(xmp)
    if m:
        s = m.group(1).decode(errors="ignore")
        fm = _FLOAT_RE.search(s)
        return float(fm.group()) if fm else None
    return None


def read_frame_meta_from_jpg(jpg_path: str) -> FrameMeta:
    with open(jpg_path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    if "GPS GPSLatitude" not in tags or "GPS GPSLongitude" not in tags:
        raise RuntimeError(f"No GPS data in {jpg_path}")

    lat = _dms_to_deg(tags["GPS GPSLatitude"], tags["GPS GPSLatitudeRef"])
    lon = _dms_to_deg(tags["GPS GPSLongitude"], tags["GPS GPSLongitudeRef"])

    alt_msl = None
    if "GPS GPSAltitude" in tags:
        a = tags["GPS GPSAltitude"].values[0]
        alt_msl = _ratio_to_float(a)

    xmp = _extract_xmp(jpg_path)

    yaw = _xmp_get_float(xmp, "Camera:Yaw")
    alt_agl = _xmp_get_float(xmp, "Camera:AboveGroundAltitude")

    if yaw is None:
        yaw = 0.0

    if alt_agl is None:
        if alt_msl is None:
            raise RuntimeError("No altitude found (XMP or EXIF)")
        alt_agl = alt_msl

    return FrameMeta(
        lat=float(lat),
        lon=float(lon),
        yaw_deg=float(yaw),
        alt_agl_m=float(alt_agl),
        alt_msl_m=(float(alt_msl) if alt_msl is not None else None),
    )


def read_frame_meta_from_tiff(tiff_path: str):
    jpg_path = find_matching_jpg(tiff_path)
    if jpg_path is None:
        return None

    try:
        return read_frame_meta_from_jpg(jpg_path)
    except Exception:
        return None


def _utm_crs(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_dict({"proj": "utm", "zone": zone, "south": lat < 0})


def altitude_for_georef_m(meta: FrameMeta) -> float:
    return float(meta.alt_msl_m) if meta.alt_msl_m is not None else float(meta.alt_agl_m)


def warn_high_altitudes(tiffs):
    high = []

    for p in tiffs:
        meta = read_frame_meta_from_tiff(p)
        if meta is None:
            continue

        alt_m = altitude_for_georef_m(meta)

        if alt_m > HIGH_ALTITUDE_WARNING_M:
            high.append((os.path.basename(p), float(alt_m)))

    if not high:
        return

    max_alt = max(a for _, a in high)

    print(
        f"WARNING: {len(high)} frame(s) have ASL/MSL altitude above "
        f"{HIGH_ALTITUDE_WARNING_M:.0f} m. Max altitude = {max_alt:.1f} m. "
        f"High altitude increases GSD; consider increasing AGG_GRID_RES_M."
    )

    for base, alt_m in high[:10]:
        print(f"  {base}: alt_used_m={alt_m:.1f}")

    if len(high) > 10:
        print(f"  ... {len(high) - 10} more")


def footprint_dims_m_from_meta(meta: FrameMeta):
    alt_m = altitude_for_georef_m(meta)

    hfov = math.radians(THERMAL_HFOV_DEG)
    vfov = math.radians(THERMAL_VFOV_DEG)

    ground_w_m = 2 * alt_m * math.tan(hfov / 2)
    ground_h_m = 2 * alt_m * math.tan(vfov / 2)

    return float(ground_w_m), float(ground_h_m), float(alt_m)


def thermal_pixel_to_lonlat(
    x_px: float,
    y_px: float,
    meta: FrameMeta,
) -> Tuple[float, float]:
    ground_w_m, ground_h_m, _ = footprint_dims_m_from_meta(meta)

    mx = ground_w_m / THERMAL_WIDTH_PX
    my = ground_h_m / THERMAL_HEIGHT_PX

    cx = (THERMAL_WIDTH_PX - 1) / 2
    cy = (THERMAL_HEIGHT_PX - 1) / 2

    dx_m = (x_px - cx) * mx
    dy_m = (y_px - cy) * my

    yaw = math.radians(meta.yaw_deg)

    east_m  =  dx_m * math.cos(yaw) + dy_m * math.sin(yaw)
    north_m = -dx_m * math.sin(yaw) + dy_m * math.cos(yaw)

    utm = _utm_crs(meta.lon, meta.lat)
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    to_wgs = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)

    cx_m, cy_m = to_utm.transform(meta.lon, meta.lat)
    lon, lat = to_wgs.transform(cx_m + east_m, cy_m + north_m)

    return lon, lat


def frame_footprint_lonlat(tiff_path: str):
    meta = read_frame_meta_from_tiff(tiff_path)
    if meta is None:
        return None, None

    pts = [
        (0, 0),
        (THERMAL_WIDTH_PX - 1, 0),
        (THERMAL_WIDTH_PX - 1, THERMAL_HEIGHT_PX - 1),
        (0, THERMAL_HEIGHT_PX - 1),
    ]

    coords = []

    try:
        for x_px, y_px in pts:
            lon, lat = thermal_pixel_to_lonlat(x_px, y_px, meta)
            coords.append((float(lon), float(lat)))
    except Exception:
        return None, meta

    coords.append(coords[0])
    return coords, meta


def save_frame_footprints_geojson(tiffs, out_path):
    features = []

    for p in tiffs:
        frame_id = find_frame_id(p)
        base = os.path.basename(p)

        coords, meta = frame_footprint_lonlat(p)
        if coords is None or meta is None:
            continue

        ground_w_m, ground_h_m, alt_used_m = footprint_dims_m_from_meta(meta)

        features.append({
            "type": "Feature",
            "properties": {
                "frame_id": frame_id,
                "file": base,
                "lat": float(meta.lat),
                "lon": float(meta.lon),
                "yaw_deg": float(meta.yaw_deg),
                "alt_agl_m": float(meta.alt_agl_m),
                "alt_msl_m": None if meta.alt_msl_m is None else float(meta.alt_msl_m),
                "alt_used_m": float(alt_used_m),
                "altitude_mode": "MSL_ASL_if_available_else_AGL",
                "thermal_hfov_deg": float(THERMAL_HFOV_DEG),
                "thermal_vfov_deg": float(THERMAL_VFOV_DEG),
                "thermal_width_px": int(THERMAL_WIDTH_PX),
                "thermal_height_px": int(THERMAL_HEIGHT_PX),
                "ground_w_m": float(ground_w_m),
                "ground_h_m": float(ground_h_m),
                "gsd_x_m": float(ground_w_m / THERMAL_WIDTH_PX),
                "gsd_y_m": float(ground_h_m / THERMAL_HEIGHT_PX),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
        })

    fc = {
        "type": "FeatureCollection",
        "features": features,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=2)


def save_total_flight_footprint_geojson(tiffs, out_path):
    utm_crs = utm_crs_for_tiffs(tiffs)
    if utm_crs is None:
        print("  WARNING: could not save flight footprint; no usable UTM CRS.")
        return

    footprint_polys = []

    for p in tiffs:
        coords_utm = frame_footprint_utm(p, utm_crs)
        if coords_utm is None:
            continue

        poly = Polygon(coords_utm)
        if not poly.is_valid:
            poly = poly.buffer(0)

        if not poly.is_empty:
            footprint_polys.append(poly)

    if not footprint_polys:
        print("  WARNING: could not save flight footprint; no valid footprints.")
        return

    union_utm = unary_union(footprint_polys)

    to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    def _to_lonlat(x, y, z=None):
        return to_wgs.transform(x, y)

    union_wgs = shapely_transform(_to_lonlat, union_utm)

    feature = {
        "type": "Feature",
        "properties": {
            "num_frames": int(len(tiffs)),
            "num_footprints_used": int(len(footprint_polys)),
            "crs_source": str(utm_crs),
            "altitude_mode": "MSL_ASL_if_available_else_AGL",
        },
        "geometry": mapping(union_wgs),
    }

    fc = {
        "type": "FeatureCollection",
        "features": [feature],
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=2)


# -------------------------
# RASTER MASK AGGREGATION HELPERS
# -------------------------


def utm_crs_for_tiffs(tiffs):
    for p in tiffs:
        meta = read_frame_meta_from_tiff(p)
        if meta is not None:
            return _utm_crs(meta.lon, meta.lat)
    return None


def frame_footprint_utm(tiff_path: str, utm_crs):
    meta = read_frame_meta_from_tiff(tiff_path)
    if meta is None:
        return None

    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

    pts = [
        (0, 0),
        (THERMAL_WIDTH_PX - 1, 0),
        (THERMAL_WIDTH_PX - 1, THERMAL_HEIGHT_PX - 1),
        (0, THERMAL_HEIGHT_PX - 1),
    ]

    coords = []

    try:
        for x_px, y_px in pts:
            lon, lat = thermal_pixel_to_lonlat(x_px, y_px, meta)
            x_m, y_m = to_utm.transform(lon, lat)
            coords.append((float(x_m), float(y_m)))
    except Exception:
        return None

    return coords


def aggregate_bounds_from_tiffs(tiffs, utm_crs):
    xs = []
    ys = []

    for p in tiffs:
        coords = frame_footprint_utm(p, utm_crs)
        if coords is None:
            continue

        for x, y in coords:
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        return None

    return min(xs), min(ys), max(xs), max(ys)


def make_aggregate_grid(tiffs, res_m):
    utm_crs = utm_crs_for_tiffs(tiffs)
    if utm_crs is None:
        return None

    bounds = aggregate_bounds_from_tiffs(tiffs, utm_crs)
    if bounds is None:
        return None

    min_x, min_y, max_x, max_y = bounds

    pad = 5.0 * res_m
    min_x -= pad
    min_y -= pad
    max_x += pad
    max_y += pad

    width = int(np.ceil((max_x - min_x) / res_m))
    height = int(np.ceil((max_y - min_y) / res_m))

    transform = from_origin(min_x, max_y, res_m, res_m)

    return {
        "crs": utm_crs,
        "transform": transform,
        "min_x": float(min_x),
        "max_y": float(max_y),
        "width": int(width),
        "height": int(height),
        "res_m": float(res_m),
    }


def init_mask_aggregator(tiffs):
    grid = make_aggregate_grid(tiffs, AGG_GRID_RES_M)
    if grid is None:
        return None

    return {
        "grid": grid,
        "s2_count": np.zeros((grid["height"], grid["width"]), dtype=np.uint16),
        "cold_counts": {
            float(delta_c): np.zeros((grid["height"], grid["width"]), dtype=np.uint16)
            for delta_c in cold_threshold_values()
        },
        "num_frames_added": 0,
        "num_frames_skipped": 0,
    }


def pixel_centers_to_agg_indices(tiff_path: str, mask_shape, agg):
    meta = read_frame_meta_from_tiff(tiff_path)
    if meta is None or agg is None:
        return None, None, None

    h, w = mask_shape
    grid = agg["grid"]

    to_utm = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True)
    center_x_m, center_y_m = to_utm.transform(meta.lon, meta.lat)

    ground_w_m, ground_h_m, _ = footprint_dims_m_from_meta(meta)

    mx = ground_w_m / THERMAL_WIDTH_PX
    my = ground_h_m / THERMAL_HEIGHT_PX

    cx = (THERMAL_WIDTH_PX - 1) / 2.0
    cy = (THERMAL_HEIGHT_PX - 1) / 2.0

    rows = np.arange(h, dtype=np.float64)
    cols = np.arange(w, dtype=np.float64)
    col_grid, row_grid = np.meshgrid(cols, rows)

    dx_m = (col_grid - cx) * mx
    dy_m = (row_grid - cy) * my

    yaw = math.radians(meta.yaw_deg)

    east_m = dx_m * math.cos(yaw) + dy_m * math.sin(yaw)
    north_m = -dx_m * math.sin(yaw) + dy_m * math.cos(yaw)

    xs = center_x_m + east_m
    ys = center_y_m + north_m

    agg_cols = np.floor((xs - grid["min_x"]) / grid["res_m"]).astype(np.int32)
    agg_rows = np.floor((grid["max_y"] - ys) / grid["res_m"]).astype(np.int32)

    in_bounds = (
        (agg_rows >= 0) &
        (agg_rows < grid["height"]) &
        (agg_cols >= 0) &
        (agg_cols < grid["width"])
    )

    return agg_rows, agg_cols, in_bounds


def add_masks_to_aggregator(agg, tiff_path, s2_coverage_mask, cumulative_cold_masks):
    if agg is None:
        return False

    if s2_coverage_mask is None or cumulative_cold_masks is None:
        agg["num_frames_skipped"] += 1
        return False

    agg_rows, agg_cols, in_bounds = pixel_centers_to_agg_indices(
        tiff_path,
        s2_coverage_mask.shape,
        agg
    )

    if agg_rows is None:
        agg["num_frames_skipped"] += 1
        return False

    width = agg["grid"]["width"]

    # S2 coverage: one vote per frame per aggregate cell
    s2_valid = s2_coverage_mask & in_bounds

    if np.any(s2_valid):
        s2_linear = agg_rows[s2_valid] * width + agg_cols[s2_valid]
        s2_unique = np.unique(s2_linear)
        agg["s2_count"].ravel()[s2_unique] += 1

    # Cold masks: one vote per frame per aggregate cell per threshold
    for delta_c, mask in cumulative_cold_masks.items():
        delta_c = float(delta_c)

        if delta_c not in agg["cold_counts"]:
            continue

        cold_valid = mask & in_bounds

        if not np.any(cold_valid):
            continue

        cold_linear = agg_rows[cold_valid] * width + agg_cols[cold_valid]
        cold_unique = np.unique(cold_linear)
        agg["cold_counts"][delta_c].ravel()[cold_unique] += 1

    agg["num_frames_added"] += 1
    return True


def write_geotiff(path, array, grid, dtype=None, nodata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    arr = array
    if dtype is not None:
        arr = arr.astype(dtype)

    profile = {
        "driver": "GTiff",
        "height": int(grid["height"]),
        "width": int(grid["width"]),
        "count": 1,
        "dtype": arr.dtype,
        "crs": grid["crs"],
        "transform": grid["transform"],
        "compress": "deflate",
    }

    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def save_aggregate_rasters(mask_agg):
    if mask_agg is None:
        return

    grid = mask_agg["grid"]
    s2_count = mask_agg["s2_count"]

    write_geotiff(
        os.path.join(RASTER_LAYER_DIR, "s2_count.tif"),
        s2_count,
        grid,
        dtype=np.uint16,
        nodata=0
    )

    valid_s2 = s2_count >= MIN_S2_SUPPORT_COUNT

    for delta_c, cold_count in mask_agg["cold_counts"].items():
        label = threshold_label(delta_c)

        cold_count_path = os.path.join(
            RASTER_LAYER_DIR,
            f"cold_{label}_count.tif"
        )

        fraction_path = os.path.join(
            RASTER_LAYER_DIR,
            f"cold_{label}_fraction.tif"
        )

        final_mask_path = os.path.join(
            RASTER_LAYER_DIR,
            f"cold_{label}_final_mask.tif"
        )

        write_geotiff(
            cold_count_path,
            cold_count,
            grid,
            dtype=np.uint16,
            nodata=0
        )

        fraction = np.full(s2_count.shape, np.nan, dtype=np.float32)
        ok = s2_count > 0
        fraction[ok] = cold_count[ok].astype(np.float32) / s2_count[ok].astype(np.float32)

        write_geotiff(
            fraction_path,
            fraction,
            grid,
            dtype=np.float32,
            nodata=np.nan
        )

        final_mask = (
            valid_s2
            & np.isfinite(fraction)
            & (fraction >= MIN_SUPPORT_FRACTION)
        )

        write_geotiff(
            final_mask_path,
            final_mask.astype(np.uint8),
            grid,
            dtype=np.uint8,
            nodata=0
        )


def summarize_aggregate_masks(mask_agg):
    if mask_agg is None:
        return {}

    grid = mask_agg["grid"]
    cell_area_m2 = float(grid["res_m"] ** 2)

    s2_count = mask_agg["s2_count"]
    observed = s2_count > 0
    valid_s2 = s2_count >= MIN_S2_SUPPORT_COUNT

    summary = {
        "cell_area_m2": cell_area_m2,
        "observed_cell_count": int(np.count_nonzero(observed)),
        "observed_area_m2": float(np.count_nonzero(observed) * cell_area_m2),
        "valid_s2_cell_count": int(np.count_nonzero(valid_s2)),
        "valid_s2_area_m2": float(np.count_nonzero(valid_s2) * cell_area_m2),
        "max_s2_count": int(np.max(s2_count)) if s2_count.size else 0,
        "thresholds": {},
    }

    for delta_c, cold_count in mask_agg["cold_counts"].items():
        label = threshold_label(delta_c)

        fraction = np.full(s2_count.shape, np.nan, dtype=np.float32)
        ok = s2_count > 0
        fraction[ok] = cold_count[ok].astype(np.float32) / s2_count[ok].astype(np.float32)

        final_mask = (
            valid_s2
            & np.isfinite(fraction)
            & (fraction >= MIN_SUPPORT_FRACTION)
        )

        cold_observed = cold_count > 0

        summary["thresholds"][label] = {
            "delta_c": float(delta_c),
            "cold_observation_count": int(np.sum(cold_count)),
            "cold_observed_cell_count": int(np.count_nonzero(cold_observed)),
            "cold_observed_area_m2": float(np.count_nonzero(cold_observed) * cell_area_m2),
            "final_cell_count": int(np.count_nonzero(final_mask)),
            "final_area_m2": float(np.count_nonzero(final_mask) * cell_area_m2),
            "max_cold_count": int(np.max(cold_count)) if cold_count.size else 0,
            "mean_support_fraction_observed": (
                float(np.nanmean(fraction[observed])) if np.any(observed) else None
            ),
            "max_support_fraction": (
                float(np.nanmax(fraction)) if np.any(np.isfinite(fraction)) else None
            ),
        }

    return summary


# -------------------------
# POLYGON CREATION HELPERS 
# -------------------------


def mask_to_union_polygon_wgs(mask, grid, min_area_m2=0.0):
    mask_u8 = mask.astype(np.uint8)

    polys_utm = []

    for geom, value in shapes(
        mask_u8,
        mask=mask_u8.astype(bool),
        transform=grid["transform"]
    ):
        if int(value) != 1:
            continue

        poly = shape(geom)

        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty:
            continue

        if min_area_m2 > 0.0 and poly.area < min_area_m2:
            continue

        polys_utm.append(poly)

    if not polys_utm:
        return None, 0, 0.0

    union_utm = unary_union(polys_utm)

    if not union_utm.is_valid:
        union_utm = union_utm.buffer(0)

    area_m2 = float(union_utm.area)

    to_wgs = Transformer.from_crs(grid["crs"], "EPSG:4326", always_xy=True)

    def _to_lonlat(x, y, z=None):
        return to_wgs.transform(x, y)

    union_wgs = shapely_transform(_to_lonlat, union_utm)

    return union_wgs, len(polys_utm), area_m2


def write_polygon_geojson(path, geometry, properties):
    if geometry is None or geometry.is_empty:
        fc = {
            "type": "FeatureCollection",
            "features": [],
        }
    else:
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": mapping(geometry),
                }
            ],
        }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(fc, f, indent=2)


def save_aggregate_polygons(mask_agg):
    if mask_agg is None:
        return {}

    grid = mask_agg["grid"]
    s2_count = mask_agg["s2_count"]

    polygon_summary = {
        "min_polygon_area_m2": float(MIN_POLYGON_AREA_M2),
        "layers": {},
    }

    valid_s2 = s2_count >= MIN_S2_SUPPORT_COUNT

    s2_geom, s2_region_count, s2_area_m2 = mask_to_union_polygon_wgs(
        valid_s2,
        grid,
        min_area_m2=MIN_POLYGON_AREA_M2
    )

    s2_path = os.path.join(POLYGON_LAYER_DIR, "s2_coverage_valid.geojson")

    write_polygon_geojson(
        s2_path,
        s2_geom,
        {
            "layer": "s2_coverage_valid",
            "min_s2_support_count": int(MIN_S2_SUPPORT_COUNT),
            "region_count_before_union": int(s2_region_count),
            "area_m2": float(s2_area_m2),
        }
    )

    polygon_summary["layers"]["s2_coverage_valid"] = {
        "path": s2_path,
        "region_count_before_union": int(s2_region_count),
        "area_m2": float(s2_area_m2),
    }

    for delta_c, cold_count in mask_agg["cold_counts"].items():
        label = threshold_label(delta_c)

        fraction = np.full(s2_count.shape, np.nan, dtype=np.float32)
        ok = s2_count > 0
        fraction[ok] = cold_count[ok].astype(np.float32) / s2_count[ok].astype(np.float32)

        final_mask = (
            valid_s2
            & np.isfinite(fraction)
            & (fraction >= MIN_SUPPORT_FRACTION)
        )

        geom, region_count, area_m2 = mask_to_union_polygon_wgs(
            final_mask,
            grid,
            min_area_m2=MIN_POLYGON_AREA_M2
        )

        out_path = os.path.join(
            POLYGON_LAYER_DIR,
            f"cold_{label}_final.geojson"
        )

        write_polygon_geojson(
            out_path,
            geom,
            {
                "layer": f"cold_{label}_final",
                "delta_c": float(delta_c),
                "min_support_fraction": float(MIN_SUPPORT_FRACTION),
                "min_s2_support_count": int(MIN_S2_SUPPORT_COUNT),
                "region_count_before_union": int(region_count),
                "area_m2": float(area_m2),
            }
        )

        polygon_summary["layers"][f"cold_{label}_final"] = {
            "path": out_path,
            "delta_c": float(delta_c),
            "region_count_before_union": int(region_count),
            "area_m2": float(area_m2),
        }

    return polygon_summary


# -------------------------
# BIAS FIELD YAW GROUPING HELPERS 
# -------------------------


def read_yaw_from_jpg(jpg_path: str):
    if jpg_path is None:
        return None

    try:
        with open(jpg_path, "rb") as f:
            data = f.read()
    except OSError:
        return None

    m = re.search(rb'Camera:Yaw="([^"]+)"', data)
    if not m:
        return None

    try:
        return float(m.group(1).decode(errors="ignore"))
    except ValueError:
        return None


YAW_PEAK_DEGREES = []


def circular_distance_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def find_yaw_histogram_peaks(yaws_deg):
    yaws = np.asarray(yaws_deg, dtype=np.float64)
    yaws = yaws[np.isfinite(yaws)] % 360.0

    if yaws.size == 0:
        return [], None, None, None

    edges = np.arange(0.0, 360.0 + YAW_HIST_BIN_DEG, YAW_HIST_BIN_DEG)
    hist, edges = np.histogram(yaws, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0

    hs = gaussian_filter1d(
        hist.astype(np.float64),
        sigma=YAW_HIST_SMOOTH_SIGMA_BINS,
        mode="wrap"
    )

    if np.max(hs) <= 0:
        return [], hist, hs, centers

    min_distance_bins = max(1, int(round(YAW_PEAK_MIN_DISTANCE_DEG / YAW_HIST_BIN_DEG)))

    candidate_peaks, _ = find_peaks(
        hs,
        distance=min_distance_bins
    )

    kept_peaks = []
    support_half_width = float(YAW_PEAK_SUPPORT_WINDOW_DEG)

    for i in candidate_peaks:
        peak_deg = float(centers[i] % 360.0)

        raw_support = 0
        for c, h in zip(centers, hist):
            if circular_distance_deg(float(c), peak_deg) <= support_half_width:
                raw_support += int(h)

        if raw_support >= YAW_PEAK_MIN_RAW_COUNT:
            kept_peaks.append(peak_deg)

    kept_peaks = sorted(kept_peaks)

    return kept_peaks, hist, hs, centers


def yaw_to_nearest_peak_bin(yaw_deg):
    if yaw_deg is None or not YAW_PEAK_DEGREES:
        return None

    yaw = yaw_deg % 360.0
    nearest = min(YAW_PEAK_DEGREES, key=lambda p: circular_distance_deg(yaw, p))

    return f"yaw_peak_{int(round(nearest)) % 360:03d}"


def frame_direction_from_tiff(tiff_path: str):
    jpg = find_matching_jpg(tiff_path)
    yaw = read_yaw_from_jpg(jpg)
    return yaw_to_nearest_peak_bin(yaw), yaw


def collect_yaws_from_tiffs(tiffs):
    yaws = []

    for p in tiffs:
        yaw = read_yaw_from_jpg(find_matching_jpg(p))
        if yaw is not None and np.isfinite(yaw):
            yaws.append(float(yaw))

    return yaws


def save_yaw_histogram_plot(hist, hs, centers, peak_degrees):
    os.makedirs(BIAS_FIELD_DIR, exist_ok=True)

    if hist is None or hs is None or centers is None:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(centers, hist, alpha=0.35, label="raw yaw histogram")
    plt.plot(centers, hs, linewidth=2, label="smoothed yaw histogram")

    ymax = float(np.max(hs)) if np.max(hs) > 0 else 1.0

    for peak in peak_degrees:
        plt.axvline(peak, linestyle="--", linewidth=1.5)
        plt.text(
            peak,
            ymax * 0.95,
            f"{peak:.0f}°",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )

    plt.title(
        f"Yaw histogram peak detection\n"
        f"bin={YAW_HIST_BIN_DEG}°, smooth_sigma={YAW_HIST_SMOOTH_SIGMA_BINS} bins, "
        f"min_dist={YAW_PEAK_MIN_DISTANCE_DEG}°, "
        f"min_raw_height={YAW_PEAK_MIN_RAW_COUNT} frames"
    )
    plt.xlabel("Yaw (degrees)")
    plt.ylabel("Count")
    plt.xlim(0, 360)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BIAS_FIELD_DIR, "yaw_histogram_peaks.png"), dpi=180)
    plt.close()


# -------------------------
# BIAS FIELD CONSTRUCTION HELPERS
# -------------------------


def local_median_fill(arr, valid_mask, size=25):
    filled = arr.copy()

    temp = np.where(valid_mask, arr, 0.0)
    valid_count = uniform_filter(valid_mask.astype(np.float32), size=size) * (size * size)
    local_med = median_filter(temp, size=size)

    use_local = (~valid_mask) & (valid_count > 0)
    filled[use_local] = local_med[use_local]

    global_med = float(np.nanmedian(arr[valid_mask])) if np.any(valid_mask) else 0.0
    filled[~np.isfinite(filled)] = global_med

    return filled


def init_bias_accumulators(shape):
    return {
        "sum": np.zeros(shape, dtype=np.float64),
        "count": np.zeros(shape, dtype=np.int32),
    }


def update_bias_accumulators(acc, temp_c, S2):
    if not np.any(S2):
        return

    frame_ref_c = float(np.nanpercentile(temp_c[S2], BASELINE_PERCENTILE))
    diff_for_bias = temp_c - frame_ref_c

    valid = S2 & np.isfinite(diff_for_bias)
    acc["sum"][valid] += diff_for_bias[valid]
    acc["count"][valid] += 1


def finalize_bias_field(acc):
    count = acc["count"]
    raw_bias = np.full(acc["sum"].shape, np.nan, dtype=np.float32)

    valid = count >= BIAS_MIN_COUNT
    raw_bias[valid] = (acc["sum"][valid] / count[valid]).astype(np.float32)

    filled = local_median_fill(raw_bias, valid, size=25)

    smooth_bias = gaussian_filter(filled, sigma=BIAS_SMOOTH_SIGMA)

    if np.any(valid):
        smooth_bias = smooth_bias - float(np.nanpercentile(smooth_bias[valid], BASELINE_PERCENTILE))

    return raw_bias.astype(np.float32), smooth_bias.astype(np.float32), count


def init_directional_bias_state(shape):
    return {
        "global": init_bias_accumulators(shape),
        "by_dir": {},
        "usable_frames_by_dir": {},
        "shape": shape,
    }


def update_directional_bias_state(state, tiff_path, temp_c, S2):
    update_bias_accumulators(state["global"], temp_c, S2)

    direction, yaw = frame_direction_from_tiff(tiff_path)

    if direction is not None:
        if direction not in state["by_dir"]:
            state["by_dir"][direction] = init_bias_accumulators(state["shape"])
            state["usable_frames_by_dir"][direction] = 0

        update_bias_accumulators(state["by_dir"][direction], temp_c, S2)
        state["usable_frames_by_dir"][direction] += 1

    return direction, yaw


def finalize_directional_bias_state(state):
    raw_global, smooth_global, count_global = finalize_bias_field(state["global"])

    out = {
        "global": {
            "raw": raw_global,
            "smooth": smooth_global,
            "count": count_global,
            "usable_frames": int(sum(state["usable_frames_by_dir"].values())),
            "fallback": False,
        },
        "by_dir": {},
        "usable_frames_by_dir": dict(state["usable_frames_by_dir"]),
    }

    for direction, acc in state["by_dir"].items():
        usable = int(state["usable_frames_by_dir"].get(direction, 0))

        raw_d, smooth_d, count_d = finalize_bias_field(acc)

        out["by_dir"][direction] = {
            "raw": raw_d,
            "smooth": smooth_d,
            "count": count_d,
            "usable_frames": usable,
            "fallback": False,
        }

    return out


def save_one_bias_field(prefix, raw_bias, smooth_bias, count, extra_title=""):
    os.makedirs(BIAS_FIELD_DIR, exist_ok=True)

    np.save(os.path.join(BIAS_FIELD_DIR, f"{prefix}_raw.npy"), raw_bias)
    np.save(os.path.join(BIAS_FIELD_DIR, f"{prefix}_smooth.npy"), smooth_bias)
    np.save(os.path.join(BIAS_FIELD_DIR, f"{prefix}_count.npy"), count)

    plt.figure(figsize=(8, 6))
    plt.imshow(smooth_bias, cmap="coolwarm")
    plt.colorbar(label="Estimated residual bias (°C)")
    plt.title(
        f"{prefix} smoothed residual bias field\n"
        f"{extra_title}\n"
        f"sigma={BIAS_SMOOTH_SIGMA}, min_count={BIAS_MIN_COUNT}"
    )
    plt.axis("off")
    plt.savefig(os.path.join(BIAS_FIELD_DIR, f"{prefix}_smooth.png"), dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(count, cmap="viridis")
    plt.colorbar(label="Number of S2 observations")
    plt.title(f"{prefix} bias field observation count")
    plt.axis("off")
    plt.savefig(os.path.join(BIAS_FIELD_DIR, f"{prefix}_count.png"), dpi=180, bbox_inches="tight")
    plt.close()


def save_directional_bias_outputs(bias_bundle):
    save_one_bias_field(
        "bias_global",
        bias_bundle["global"]["raw"],
        bias_bundle["global"]["smooth"],
        bias_bundle["global"]["count"],
        extra_title="global"
    )

    summary = {
        "use_directional_bias": bool(USE_DIRECTIONAL_BIAS),
        "apply_bias_correction": bool(APPLY_BIAS_CORRECTION),
        "yaw_hist_bin_deg": int(YAW_HIST_BIN_DEG),
        "yaw_hist_smooth_sigma_bins": float(YAW_HIST_SMOOTH_SIGMA_BINS),
        "yaw_peak_min_distance_deg": float(YAW_PEAK_MIN_DISTANCE_DEG),
        "yaw_peak_min_raw_count": int(YAW_PEAK_MIN_RAW_COUNT),
        "yaw_peak_degrees": [float(p) for p in YAW_PEAK_DEGREES],
        "usable_frames_by_dir": bias_bundle["usable_frames_by_dir"],
        "directions": {},
    }

    for direction, item in bias_bundle["by_dir"].items():
        summary["directions"][direction] = {
            "usable_frames": int(item["usable_frames"]),
            "fallback": bool(item["fallback"]),
        }

        if item["fallback"]:
            continue

        prefix = f"bias_{direction}"
        extra = f"{direction}, usable_frames={item['usable_frames']}, fallback=False"

        save_one_bias_field(
            prefix,
            item["raw"],
            item["smooth"],
            item["count"],
            extra_title=extra
        )

    with open(os.path.join(BIAS_FIELD_DIR, "bias_direction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def get_valid_bias_directions(bias_bundle):
    return {
        direction
        for direction, item in bias_bundle["by_dir"].items()
        if not item.get("fallback", False)
    }


def choose_bias_for_frame(tiff_path, bias_bundle):
    if not APPLY_BIAS_CORRECTION or bias_bundle is None:
        return None, "none"

    if USE_DIRECTIONAL_BIAS:
        direction, yaw = frame_direction_from_tiff(tiff_path)

        if direction in bias_bundle["by_dir"]:
            return bias_bundle["by_dir"][direction]["smooth"], direction

        return bias_bundle["global"]["smooth"], f"global_fallback_for_{direction}"

    return bias_bundle["global"]["smooth"], "global"


# -------------------------
# FRAME PROCESSING
# -------------------------


def process_one(tiff_path: str, report, bias_state=None, bias_field=None, bias_field_name=None, mask_agg=None, save_outputs=True):
    frame_id = find_frame_id(tiff_path)
    base = os.path.basename(tiff_path)

    with rasterio.open(tiff_path) as src:
        raw = src.read(1)
        if raw.dtype.kind not in ("u", "i", "f"):
            print(f"{base}: SKIP (unexpected dtype {raw.dtype})")
            _report_add_skip(report, frame_id, base, "unexpected_dtype", {"dtype": str(raw.dtype)}, step="read_tiff")
            return

    temp_c = to_celsius_autel(raw)
    temp_c[(temp_c < -50) | (temp_c > 200)] = np.nan
    finite = np.isfinite(temp_c)
    if not np.any(finite):
        print(f"{base}: SKIP (no finite temps)")
        _report_add_skip(report, frame_id, base, "no_finite_temps", step="temp_sanitize")
        return

    tex = local_std_texture(temp_c, finite, TEX_WIN)
    tex_vals = tex[finite]
    hist, bin_edges = texture_hist(tex_vals, nbins=TEX_BINS)
    if hist is None:
        print(f"{base}: SKIP (texture hist failed)")
        _report_add_skip(report, frame_id, base, "texture_hist_failed", step="texture_hist")
        return

    thr, angles, hs, centers = angle_knee_threshold(hist, bin_edges)

    if thr is None:
        step = "texture_hist_qc"
        reason = "no_full_angle_run"

        # optional: still save debug so you can inspect these failures
        save_cut_debug(
            frame_id, base, temp_c,
            hist, bin_edges, 0.0,
            angles, hs, centers,
            step, reason
        )

        _report_add_skip(
            report, frame_id, base, reason,
            {
                "angle_deg": float(ANGLE_DEG),
                "angle_run": int(ANGLE_RUN),
            },
            step=step
        )
        return

    hist_metrics = compute_hist_metrics(hist, hs, centers, thr)

    # Filters

    # Primary
    if FILTERS_ENABLED.get("peak_height_min", False) and (hist_metrics["rawer_peak_height"] < PEAK_HEIGHT_MIN):
        step = "hist_metric_qc"
        reason = "peak_height_below_min"
        details = dict(hist_metrics)
        details["peak_height_min"] = float(PEAK_HEIGHT_MIN)

        save_cut_debug(
            frame_id, base, temp_c,
            hist, bin_edges, thr,
            angles, hs, centers,
            step, reason
        )

        _report_add_skip(report, frame_id, base, reason, details, step=step)
        return

    # Liberal sanity check options

    if FILTERS_ENABLED.get("thr_max", False) and (thr > TEXTURE_THR_MAX):
        step = "texture_hist_qc"
        reason = "texture_thr_above_max"

        details = dict(hist_metrics)
        details["thr_max"] = float(TEXTURE_THR_MAX)

        save_cut_debug(
            frame_id, base, temp_c,
            hist, bin_edges, thr,
            angles, hs, centers,
            step, reason
        )

        _report_add_skip(report, frame_id, base, reason, details, step=step)
        return
    
    if FILTERS_ENABLED.get("peak_x_max", False) and (hist_metrics["peak_x"] > PEAK_X_MAX):
        step = "hist_metric_qc"
        reason = "peak_x_above_max"
        details = dict(hist_metrics)
        details["peak_x_max"] = float(PEAK_X_MAX)

        save_cut_debug(
            frame_id, base, temp_c,
            hist, bin_edges, thr,
            angles, hs, centers,
            step, reason
        )

        _report_add_skip(report, frame_id, base, reason, details, step=step)
        return

    if FILTERS_ENABLED.get("peak_width_max", False) and (hist_metrics["peak_width_fwhm"] > PEAK_WIDTH_MAX):
        step = "hist_metric_qc"
        reason = "peak_width_above_max"
        details = dict(hist_metrics)
        details["peak_width_max"] = float(PEAK_WIDTH_MAX)

        save_cut_debug(
            frame_id, base, temp_c,
            hist, bin_edges, thr,
            angles, hs, centers,
            step, reason
        )

        _report_add_skip(report, frame_id, base, reason, details, step=step)
        return

    if FILTERS_ENABLED.get("peak_sharpness_min", False) and (hist_metrics["peak_sharpness"] < PEAK_SHARPNESS_MIN):
        step = "hist_metric_qc"
        reason = "peak_sharpness_below_min"
        details = dict(hist_metrics)
        details["peak_sharpness_min"] = float(PEAK_SHARPNESS_MIN)

        save_cut_debug(
            frame_id, base, temp_c,
            hist, bin_edges, thr,
            angles, hs, centers,
            step, reason
        )

        _report_add_skip(report, frame_id, base, reason, details, step=step)
        return


    S = finite & (tex <= thr)
    S2, dist, L = build_s2(S)

    frac_finite = float(np.count_nonzero(finite))
    s_frac = float(np.count_nonzero(S)) / frac_finite
    s2_frac = float(np.count_nonzero(S2)) / frac_finite
    ratio = s2_frac / s_frac if s_frac > 0 else 0.0

    if FILTERS_ENABLED.get("s2_over_s_min", False) and (ratio < S2_OVER_S_MIN):
        step = "segmentation_qc"
        reason = "s2_over_s_below_min"

        details = dict(hist_metrics)
        details.update({
            "s_frac": float(s_frac),
            "s2_frac": float(s2_frac),
            "s2_over_s": float(ratio),
            "s2_over_s_min": float(S2_OVER_S_MIN),
        })

        save_cut_debug(
            frame_id, base, temp_c,
            hist, bin_edges, thr,
            angles, hs, centers,
            step, reason
        )

        _report_add_skip(report, frame_id, base, reason, details, step=step)
        return

    bias_direction = None
    bias_yaw_deg = None

    if bias_state is not None:
        if not (FILTERS_ENABLED.get("s2_min_frac_baseline_skip", True) and (s2_frac < MIN_SMOOTH_FRAC)):
            bias_direction, bias_yaw_deg = update_directional_bias_state(
                bias_state,
                tiff_path,
                temp_c,
                S2
            )

        if not save_outputs:
            return

    if save_outputs and frame_id:
        save_hist_angle_debug(frame_id, base, hist, bin_edges, thr, angles, hs, centers)

    print(base)
    print(f"  angle_knee_thr(texture)={thr:.6f}")
    print(f"  S  frac={s_frac:.4f}")
    if S2_MODE == "within_x":
        print(f"  S2 frac={s2_frac:.4f}  (mode=within_x, x={DILATE_PIXELS}px, conn={'8' if CONNECTIVITY_8 else '4'})")
    else:
        print(f"  S2 frac={s2_frac:.4f}  (mode={S2_MODE}, conn={'8' if CONNECTIVITY_8 else '4'})")

    analysis_temp_c = temp_c
    bias_correction_applied = False
    if APPLY_BIAS_CORRECTION and bias_field is not None:
        analysis_temp_c = temp_c - bias_field
        bias_correction_applied = True
    if bias_field_name is None:
        bias_field_name = "none"
    

    baseline_c = None
    if FILTERS_ENABLED.get("s2_min_frac_baseline_skip", True) and (s2_frac < MIN_SMOOTH_FRAC):
        print(f"  BASELINE(S2) SKIP (S2 frac < {MIN_SMOOTH_FRAC:.2f})\n")
        
        save_cut_debug(
        frame_id, base, temp_c,
        hist, bin_edges, thr,
        angles, hs, centers,
        step="baseline",
        reason="s2_frac_below_min"
        )

        details = dict(hist_metrics)
        details.update({
            "s_frac": float(s_frac),
            "s2_frac": float(s2_frac),
            "s2_over_s": float(ratio),
            "min_smooth_frac": float(MIN_SMOOTH_FRAC),
        })

        _report_add_baseline_skip(
            report, frame_id, base,
            "s2_frac_below_min",
            details,
            step="baseline"
        )
    else:
        baseline_c = float(np.nanpercentile(analysis_temp_c[S2], BASELINE_PERCENTILE))
        print(f"  BASELINE(S2) = p{BASELINE_PERCENTILE} = {baseline_c:.3f} °C\n")

    disp_gray = normalize_for_display(analysis_temp_c)

    outS = _pass_path(f"IRX_{frame_id}_S_on_gray.png")
    overlay_and_save(disp_gray, S, f"{base}\nS: tex<=thr  frac={s_frac:.3f}", outS)

    outS2 = _pass_path(f"IRX_{frame_id}_S2_on_gray.png")
    
    if S2_MODE == "within_x":
        title = f"{base}\nS2: within {DILATE_PIXELS}px of largest  frac={s2_frac:.3f}"
    else:
        title = f"{base}\nS2: contiguous ({S2_MODE})  frac={s2_frac:.3f}"
    overlay_and_save(disp_gray, S2, title, outS2)

    if SAVE_DIST_HEATMAP and S2_MODE == "within_x":
        save_dist_heatmap(frame_id, base, disp_gray, dist, L)

    diff_c = None
    contour_min_delta_c = None
    contour_levels_c = None
    s2_coverage_mask = None
    cumulative_cold_masks = {}

    if baseline_c is not None:
        diff_c = analysis_temp_c - baseline_c

        if BUILD_CUMULATIVE_COLD_MASKS:
            s2_coverage_mask, cumulative_cold_masks = build_cumulative_cold_masks(diff_c, S2)

        s2_valid = S2 & np.isfinite(diff_c)
        s2_diff = diff_c[s2_valid]

        if s2_diff.size > 0:
            # Only cooler-than-baseline depth matters for contours
            cool_depth = np.maximum(-s2_diff, 0.0)
            coolest_depth_c = float(np.max(cool_depth))
            contour_min_delta_c = -coolest_depth_c

            n_steps = max(1, int(np.ceil(coolest_depth_c / CONTOUR_STEP_C)))
            contour_levels_c = [round(-i * CONTOUR_STEP_C, 2) for i in range(n_steps + 1)]

            # Bin all S2 pixels by 0.25 C cooling depth from baseline
            full_cool_depth = np.zeros(diff_c.shape, dtype=np.float32)
            full_cool_depth[s2_valid] = np.maximum(-diff_c[s2_valid], 0.0)

            contour_idx = np.full(diff_c.shape, -1, dtype=np.int32)
            contour_idx[s2_valid] = np.floor(full_cool_depth[s2_valid] / CONTOUR_STEP_C).astype(np.int32)
            contour_idx[s2_valid] = np.clip(contour_idx[s2_valid], 0, n_steps)

            plt.figure(figsize=(8, 6))
            plt.imshow(disp_gray, cmap="gray", vmin=0.0, vmax=1.0)

            overlay = np.zeros((diff_c.shape[0], diff_c.shape[1], 4), dtype=np.float32)

            valid = contour_idx >= 0
            if np.any(valid):
                frac = contour_idx[valid].astype(np.float32) / max(n_steps, 1)

                # red at baseline, blue at coolest
                overlay[..., 0][valid] = 1.0 - frac
                overlay[..., 2][valid] = frac
                overlay[..., 3][valid] = 0.55

            plt.imshow(overlay)

            plt.title(
                f"{base}\n"
                f"Bias field: {bias_field_name}\n"
                f"S2 contours relative to baseline p{BASELINE_PERCENTILE}={baseline_c:.2f}°C "
                f"(step={CONTOUR_STEP_C:.2f}°C, coolest={contour_min_delta_c:.2f}°C)"
            )
            plt.axis("off")
            plt.savefig(
                _pass_path(f"IRX_{frame_id}_s2_temp_contours_on_gray.png"),
                dpi=150,
                bbox_inches="tight"
            )
            plt.close()

        if BUILD_AGGREGATE_MASK_LAYERS and mask_agg is not None:
            add_masks_to_aggregator(
                mask_agg,
                tiff_path,
                s2_coverage_mask,
                cumulative_cold_masks
            )

    cold_mask_pixel_counts = {}
    if baseline_c is not None and BUILD_CUMULATIVE_COLD_MASKS:
        for delta_c, mask in cumulative_cold_masks.items():
            cold_mask_pixel_counts[threshold_label(delta_c)] = int(np.count_nonzero(mask))

    details = dict(hist_metrics)
    details.update({
        # texture / histogram thresholding
        "tex_win": int(TEX_WIN),
        "hist_smooth_k": int(HIST_SMOOTH_K),
        "angle_deg": float(ANGLE_DEG),
        "angle_run": int(ANGLE_RUN),

        # S / S2 mask construction
        "s_frac": float(s_frac),
        "s2_frac": float(s2_frac),
        "s2_over_s": float(ratio),
        "s2_mode": str(S2_MODE),
        "dilate_pixels": int(DILATE_PIXELS),
        "connectivity_8": bool(CONNECTIVITY_8),

        # yaw peak grouping for directional bias
        "yaw_hist_bin_deg": int(YAW_HIST_BIN_DEG),
        "yaw_hist_smooth_sigma_bins": float(YAW_HIST_SMOOTH_SIGMA_BINS),
        "yaw_peak_min_distance_deg": float(YAW_PEAK_MIN_DISTANCE_DEG),
        "yaw_peak_support_window_deg": float(YAW_PEAK_SUPPORT_WINDOW_DEG),
        "yaw_peak_min_raw_count": int(YAW_PEAK_MIN_RAW_COUNT),
        "yaw_peak_degrees": [float(p) for p in YAW_PEAK_DEGREES],

        # bias correction
        "bias_correction_applied": bool(bias_correction_applied),
        "use_directional_bias": bool(USE_DIRECTIONAL_BIAS),
        "bias_field_name": str(bias_field_name),
        "bias_direction": None if bias_direction is None else str(bias_direction),
        "bias_yaw_deg": None if bias_yaw_deg is None else float(bias_yaw_deg),
        "bias_min_count": int(BIAS_MIN_COUNT),
        "bias_smooth_sigma": float(BIAS_SMOOTH_SIGMA),

        # baseline
        "baseline_percentile": int(BASELINE_PERCENTILE),
        "baseline_c": None if baseline_c is None else float(baseline_c),

        # visual contour output
        "contour_step_c": float(CONTOUR_STEP_C),
        "contour_min_delta_c": None if contour_min_delta_c is None else float(contour_min_delta_c),
        "contour_levels_c": contour_levels_c,

        # cumulative cold masks for later aggregation
        "build_cumulative_cold_masks": bool(BUILD_CUMULATIVE_COLD_MASKS),
        "cold_mask_min_delta_c": float(COLD_MASK_MIN_DELTA_C),
        "cold_mask_max_delta_c": float(COLD_MASK_MAX_DELTA_C),
        "cold_mask_thresholds_c": [float(x) for x in cold_threshold_values()],
        "cold_mask_pixel_counts": cold_mask_pixel_counts,

        # aggregate mask layers
        "build_aggregate_mask_layers": bool(BUILD_AGGREGATE_MASK_LAYERS),
        "agg_grid_res_m": float(AGG_GRID_RES_M),
        "min_support_fraction": float(MIN_SUPPORT_FRACTION),
        "min_s2_support_count": int(MIN_S2_SUPPORT_COUNT),
    })

    if baseline_c is not None:
        _report_add_processed(report, frame_id, base, details)


# -------------------------
# MAIN DRIVER
# -------------------------


def main(flight_root: str):
    report = _new_report()

    tiffs, media_dirs = list_camera_tiffs(flight_root)

    if not tiffs:
        print(f"No camera TIFFs found under: {flight_root}")
        if media_dirs:
            print(f"Found {len(media_dirs)} ###MEDIA dirs (example):")
            for d in media_dirs[:5]:
                print(" ", d)
        else:
            print("No ###MEDIA dirs found; searched entire tree.")
        any_irx = glob.glob(os.path.join(flight_root, "**", "IRX_*"), recursive=True)
        if any_irx:
            print("Found IRX_* paths (example):")
            for p in sorted(any_irx)[:10]:
                print(" ", os.path.basename(p))
        raise SystemExit("Stopping: adjust discovery logic to match your actual filenames.")

    selected = pick_bursts(tiffs, BURST_SIZE, NUM_BURSTS)

    global YAW_PEAK_DEGREES

    all_yaws = collect_yaws_from_tiffs(tiffs)
    YAW_PEAK_DEGREES, yaw_hist, yaw_hs, yaw_centers = find_yaw_histogram_peaks(all_yaws)
    save_yaw_histogram_plot(yaw_hist, yaw_hs, yaw_centers, YAW_PEAK_DEGREES)

    print(f"Detected yaw peak(s): {[round(p, 2) for p in YAW_PEAK_DEGREES]}")

    print(f"Selected frames ({NUM_BURSTS} burst(s) of {BURST_SIZE}):")
    for p in selected:
        print(" ", os.path.basename(p))
    print("")

    warn_high_altitudes(tiffs)

    print("Saving total flight footprint...")
    save_total_flight_footprint_geojson(
        tiffs,
        os.path.join(LAYER_DIR, "flight_footprint.geojson")
    )

    mask_agg = None
    if BUILD_AGGREGATE_MASK_LAYERS:
        print("Initializing aggregate mask grid...")
        mask_agg = init_mask_aggregator(tiffs)

        if mask_agg is None:
            print("  WARNING: aggregate mask grid could not be initialized.")
        else:
            grid = mask_agg["grid"]
            print(
                f"  aggregate grid: {grid['width']} x {grid['height']} "
                f"cells at {grid['res_m']} m"
            )

    print(f"Building bias field from all {len(tiffs)} frame(s)...")
    bias_report = _new_report()
    bias_state = None
    global SAVE_CUT_DEBUG
    _orig_save_cut_debug = SAVE_CUT_DEBUG
    SAVE_CUT_DEBUG = False
    for p in tiffs:
        if bias_state is None:
            with rasterio.open(p) as src:
                shape = src.read(1).shape
            bias_state = init_directional_bias_state(shape)

        process_one(
            p,
            bias_report,
            bias_state=bias_state,
            bias_field=None,
            bias_field_name=None,
            save_outputs=False
        )

    SAVE_CUT_DEBUG = _orig_save_cut_debug

    bias_bundle = finalize_directional_bias_state(bias_state)
    save_directional_bias_outputs(bias_bundle)

    report = _new_report()
    report["num_selected"] = len(selected)

    for p in selected:
        frame_bias, frame_bias_name = choose_bias_for_frame(p, bias_bundle)

        process_one(
            p,
            report,
            bias_state=None,
            bias_field=frame_bias,
            bias_field_name=frame_bias_name,
            mask_agg=mask_agg,
            save_outputs=True,
        )

    if mask_agg is not None:
        print("Saving aggregate raster layers...")
        save_aggregate_rasters(mask_agg)

    aggregate_polygon_summary = {}

    if mask_agg is not None and BUILD_AGGREGATE_POLYGONS:
        print("Saving aggregate polygon layers...")
        aggregate_polygon_summary = save_aggregate_polygons(mask_agg)

    if mask_agg is not None:
        report["mask_aggregation"] = {
            "num_frames_added": int(mask_agg["num_frames_added"]),
            "num_frames_skipped": int(mask_agg["num_frames_skipped"]),
            "grid": {
                "width": int(mask_agg["grid"]["width"]),
                "height": int(mask_agg["grid"]["height"]),
                "res_m": float(mask_agg["grid"]["res_m"]),
                "crs": str(mask_agg["grid"]["crs"]),
                "min_x": float(mask_agg["grid"]["min_x"]),
                "max_y": float(mask_agg["grid"]["max_y"]),
            },
            "thresholds_c": [float(x) for x in cold_threshold_values()],
            "min_support_fraction": float(MIN_SUPPORT_FRACTION),
            "min_s2_support_count": int(MIN_S2_SUPPORT_COUNT),
        }
        
        report["aggregate_threshold_summary"] = summarize_aggregate_masks(mask_agg)
        report["aggregate_polygon_summary"] = aggregate_polygon_summary

    write_report(report, OUTPUT_ROOT)
    write_text_report(report, OUTPUT_ROOT)
    save_1d_metrics_plot(report, OUTPUT_ROOT)

    print("\n=== RUN REPORT ===")
    print(f"selected:         {report['num_selected']}")
    print(f"processed:        {report['num_processed']}")
    print(f"skipped:          {report['num_skipped']}")
    print(f"baseline_skipped: {report['num_baseline_skipped']}")
    print("skipped_by_reason:", dict(report["skipped_by_reason"]))
    print("baseline_skipped_by_reason:", dict(report["baseline_skipped_by_reason"]))
    print(f"report_json:      {os.path.join(OUTPUT_ROOT, 'run_report.json')}")
    print(f"report_txt:       {os.path.join(OUTPUT_ROOT, 'run_report.txt')}")


if __name__ == "__main__":
    FLIGHT_ROOT = '/Volumes/EXTERNAL HD/Thermal Flights/1 July 23/Tongariki Clipped'
    main(FLIGHT_ROOT)