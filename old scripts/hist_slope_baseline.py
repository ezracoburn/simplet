import os
import re
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

OUTPUT_ROOT = "/Users/ezracoburn/Documents/Simple/output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

NUM_IMAGES = 5

# Guard: require at least this fraction of pixels to be "smooth candidates" S
MIN_SMOOTH_FRAC = 0.10

# Baseline definition: p90 of temps within S (smooth pixels)
BASELINE_PERCENTILE = 90

# Scalar baseline visualization tolerance (°C)
BASELINE_BAND_DELTA_C = 0.1

# Histogram bins for texture thresholding
TEX_BINS = 256

# Angle-knee params
HIST_SMOOTH_K = 9      # moving average over histogram counts
ANGLE_DEG = 20.0       # "flattening" threshold (degrees)
ANGLE_RUN = 10          # consecutive bins that must be below ANGLE_DEG


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


def pick_evenly_spaced(items, k):
    n = len(items)
    if n <= k:
        return items
    idx = np.linspace(0, n - 1, k, dtype=int)
    return [items[i] for i in idx]


def to_celsius_autel(raw: np.ndarray) -> np.ndarray:
    # raw is Kelvin*10
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
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k, dtype=np.float64) / k
    return np.convolve(xp, ker, mode="valid")


def angle_knee_threshold(hist: np.ndarray, bin_edges: np.ndarray,
                         smooth_k: int = HIST_SMOOTH_K,
                         angle_deg: float = ANGLE_DEG,
                         run: int = ANGLE_RUN):
    """
    Threshold rule:
      - Normalize histogram to probability mass
      - Smooth
      - Compute slope dy/dx and convert to angle = arctan(slope) in degrees
      - After the peak, threshold is the first point where |angle| < angle_deg
        for `run` consecutive bins.

    Returns (thr, angles, hs, centers)
    """
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
        # fallback: first local minimum after peak
        for j in range(peak_idx + 1, len(hs) - 1):
            if hs[j - 1] > hs[j] and hs[j] <= hs[j + 1]:
                thr_idx = j
                break

    if thr_idx is None:
        thr_idx = min(peak_idx + 1, len(centers) - 1)

    return float(centers[thr_idx]), angles, hs, centers


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


def save_hist_angle_debug(frame_id: str, base: str, hist, bin_edges, thr, angles, hs, centers):
    histn = hist.astype(np.float64)
    if histn.sum() > 0:
        histn /= histn.sum()

    plt.figure(figsize=(8, 4))
    plt.plot(centers, histn, alpha=0.35, label="hist (norm)")
    plt.plot(centers, hs, label=f"smoothed (k={HIST_SMOOTH_K})")
    plt.axvline(thr, linewidth=2, label="angle-knee thr")
    plt.title(f"{base}\nAngle-knee thr={thr:.6f}  angle<{ANGLE_DEG}° run={ANGLE_RUN}")
    plt.xlabel("Texture (local std of °C)")
    plt.ylabel("Probability mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_texture_hist_angleknee.png"), dpi=160)
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
    plt.savefig(os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_texture_angle_angleknee.png"), dpi=160)
    plt.close()


def process_one(tiff_path: str):
    frame_id = find_frame_id(tiff_path)
    base = os.path.basename(tiff_path)

    with rasterio.open(tiff_path) as src:
        raw = src.read(1)
        if raw.dtype.kind not in ("u", "i", "f"):
            print(f"{base}: SKIP (unexpected dtype {raw.dtype})")
            return

    temp_c = to_celsius_autel(raw)
    temp_c[(temp_c < -50) | (temp_c > 200)] = np.nan
    finite = np.isfinite(temp_c)
    if not np.any(finite):
        print(f"{base}: SKIP (no finite temps)")
        return

    raw_valid = raw[finite]
    raw_valid = raw_valid[raw_valid > 0]
    if raw_valid.size == 0:
        print(f"{base}: SKIP (no valid raw temps > 0)")
        return

    raw_min = int(raw_valid.min())
    raw_max = int(raw_valid.max())
    min_c = raw_min * 0.1 - 273.15
    max_c = raw_max * 0.1 - 273.15

    tex = local_std_texture(temp_c, finite, win=9)
    tex_vals = tex[finite]
    hist, bin_edges = texture_hist(tex_vals, nbins=TEX_BINS)
    if hist is None:
        print(f"{base}: SKIP (texture hist failed)")
        return

    thr, angles, hs, centers = angle_knee_threshold(hist, bin_edges)

    S = finite & (tex <= thr)
    smooth_frac = float(np.count_nonzero(S)) / float(np.count_nonzero(finite))

    if frame_id:
        save_hist_angle_debug(frame_id, base, hist, bin_edges, thr, angles, hs, centers)

    print(base)
    print(f"  raw_min={raw_min}  raw_max={raw_max}")
    print(f"  temp_min={min_c:.2f}°C  temp_max={max_c:.2f}°C")
    print(f"  angle_knee_thr(texture)={thr:.6f}  smooth_frac={smooth_frac:.4f}")

    baseline_c = None
    if smooth_frac < MIN_SMOOTH_FRAC:
        print(f"  BASELINE SKIP (smooth_frac < {MIN_SMOOTH_FRAC:.2f})\n")
    else:
        baseline_c = float(np.nanpercentile(temp_c[S], BASELINE_PERCENTILE))
        print(f"  BASELINE = p{BASELINE_PERCENTILE}(S) = {baseline_c:.3f} °C\n")

    disp_gray = normalize_for_display(temp_c)

    overlay_and_save(
        disp_gray,
        S,
        f"{base}\nS: smooth (<=angle-knee thr) frac={smooth_frac:.3f}",
        os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_S_smooth_on_gray.png")
    )

    if baseline_c is not None:
        band = finite & (np.abs(temp_c - baseline_c) <= BASELINE_BAND_DELTA_C)
        overlay_and_save(
            disp_gray,
            band,
            f"{base}\nBaseline p{BASELINE_PERCENTILE}(S)={baseline_c:.2f}°C (±{BASELINE_BAND_DELTA_C}°C)",
            os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_baseline_band_on_gray.png"),
            alpha=0.55
        )


def main(flight_root: str):
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

    selected = pick_evenly_spaced(tiffs, NUM_IMAGES)
    print("Selected frames:")
    for p in selected:
        print(" ", os.path.basename(p))
    print("")

    for p in selected:
        process_one(p)


if __name__ == "__main__":
    FLIGHT_ROOT = "/Volumes/EXTERNAL HD/Thermal Flights/1 July 23/Kikirahamea - Hiva Hiva"
    main(FLIGHT_ROOT)
