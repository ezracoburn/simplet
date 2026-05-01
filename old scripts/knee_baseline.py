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
BASELINE_BAND_DELTA_C = 0.3

# Histogram bins for texture thresholding
TEX_BINS = 256

# Knee detection params (basic, robust-ish defaults)
HIST_SMOOTH_K = 9     # moving average over histogram counts
SLOPE_RUN = 8         # number of consecutive bins that must be "flat"
SLOPE_EPS_PCTL = 20   # eps = p20(|slope|) after peak


def find_frame_id(path: str):
    m = re.search(r"IRX_(\d{4})", os.path.basename(path))
    return m.group(1) if m else None


def find_media_dirs(flight_root: str):
    # Autel-style media dirs like 103MEDIA, 100MEDIA, etc.
    media_dirs = []
    for d in glob.glob(os.path.join(flight_root, "**", "*MEDIA"), recursive=True):
        if os.path.isdir(d) and re.search(r"[\\/]\d{3}MEDIA$", d):
            media_dirs.append(d)
    return sorted(set(media_dirs))


def list_camera_tiffs(flight_root: str):
    """
    Match the older scripts' behavior:
      - Prefer searching within ###MEDIA folders if present
      - Otherwise search the whole tree
      - Keep only canonical originals: IRX_####.(tif|tiff) any case
      - Exclude "weird repeats" and suffix variants
    """
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
    if background.ndim == 2:
        plt.imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
    else:
        plt.imshow(background)

    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = mask.astype(np.float32) * alpha
    plt.imshow(overlay)

    plt.title(title)
    plt.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def local_std_texture(temp_c: np.ndarray, finite: np.ndarray, win: int = 9):
    """
    Local standard deviation (texture) over a window.
    NaNs are filled with the median so we don't create NaN textures.
    """
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


def knee_threshold_from_hist(hist: np.ndarray, bin_edges: np.ndarray,
                            smooth_k: int = HIST_SMOOTH_K,
                            slope_run: int = SLOPE_RUN,
                            slope_eps_pctl: int = SLOPE_EPS_PCTL):
    """
    Basic 'knee where histogram flattens':
      - Smooth histogram counts (moving average)
      - Find peak
      - Compute slopes (diff) after peak
      - eps = p{SLOPE_EPS_PCTL}(|slope|) after peak
      - knee = first index where |slope| <= eps for SLOPE_RUN consecutive bins
    Returns (thr_value, eps)
    """
    hist = hist.astype(np.float64)
    hs = smooth_1d(hist, smooth_k)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    peak_idx = int(np.argmax(hs))
    d = np.diff(hs)
    after = d[peak_idx:]

    if after.size < slope_run + 2:
        knee_idx = min(peak_idx + 1, len(centers) - 1)
        return float(centers[knee_idx]), np.nan

    abs_after = np.abs(after)
    eps = float(np.percentile(abs_after, slope_eps_pctl))
    if eps <= 0:
        eps = float(np.max(abs_after) * 0.01) if np.max(abs_after) > 0 else 0.0

    knee_idx = None
    for i in range(peak_idx, len(d) - slope_run):
        window = d[i:i + slope_run]
        if np.all(np.abs(window) <= eps):
            knee_idx = i + 1
            break

    if knee_idx is None:
        # fallback: first local minimum after peak
        for j in range(peak_idx + 1, len(hs) - 1):
            if hs[j - 1] > hs[j] and hs[j] <= hs[j + 1]:
                knee_idx = j
                break

    if knee_idx is None:
        knee_idx = min(peak_idx + 1, len(centers) - 1)

    return float(centers[knee_idx]), eps


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


def save_hist_and_knee(frame_id: str, base: str, hist, bin_edges, thr, eps):
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hs = smooth_1d(hist.astype(np.float64), HIST_SMOOTH_K)

    plt.figure(figsize=(8, 4))
    plt.plot(centers, hist, alpha=0.35, label="hist")
    plt.plot(centers, hs, label="smoothed")
    plt.axvline(thr, linewidth=2, label="knee thr")
    plt.title(f"{base}\nKnee thr={thr:.6f}  eps(p{SLOPE_EPS_PCTL})={eps:.3g}  run={SLOPE_RUN}")
    plt.xlabel("Texture (local std of °C)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    out_hist = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_texture_hist_knee.png")
    plt.savefig(out_hist, dpi=160)
    plt.close()

    d = np.diff(hs)
    x = centers[1:]
    plt.figure(figsize=(8, 4))
    plt.plot(x, d, label="slope of smoothed hist")
    if np.isfinite(eps):
        plt.axhline(eps, linewidth=1, linestyle="--", label="+eps")
        plt.axhline(-eps, linewidth=1, linestyle="--", label="-eps")
    plt.axvline(thr, linewidth=2, label="knee thr")
    plt.title(f"{base}\nSlope after smoothing (knee when |slope| stays small)")
    plt.xlabel("Texture (local std of °C)")
    plt.ylabel("Δcount per bin")
    plt.legend()
    plt.tight_layout()
    out_slope = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_texture_slope_knee.png")
    plt.savefig(out_slope, dpi=160)
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

    thr, eps = knee_threshold_from_hist(hist, bin_edges)

    S = finite & (tex <= thr)
    smooth_frac = float(np.count_nonzero(S)) / float(np.count_nonzero(finite))

    if frame_id:
        save_hist_and_knee(frame_id, base, hist, bin_edges, thr, eps)

    print(base)
    print(f"  raw_min={raw_min}  raw_max={raw_max}")
    print(f"  temp_min={min_c:.2f}°C  temp_max={max_c:.2f}°C")
    print(f"  knee_thr(texture)={thr:.6f}  smooth_frac={smooth_frac:.4f}")

    baseline_c = None
    if smooth_frac < MIN_SMOOTH_FRAC:
        print(f"  BASELINE SKIP (smooth_frac < {MIN_SMOOTH_FRAC:.2f})\n")
    else:
        baseline_c = float(np.nanpercentile(temp_c[S], BASELINE_PERCENTILE))
        print(f"  BASELINE = p{BASELINE_PERCENTILE}(S) = {baseline_c:.3f} °C\n")

    disp_gray = normalize_for_display(temp_c)

    out1 = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_S_smooth_on_gray.png")
    overlay_and_save(
        disp_gray,
        S,
        f"{base}\nS: smooth (<=knee thr) frac={smooth_frac:.3f}",
        out1
    )

    if baseline_c is not None:
        band = finite & (np.abs(temp_c - baseline_c) <= BASELINE_BAND_DELTA_C)
        out2 = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_baseline_band_on_gray.png")
        overlay_and_save(
            disp_gray,
            band,
            f"{base}\nBaseline p{BASELINE_PERCENTILE}(S)={baseline_c:.2f}°C (±{BASELINE_BAND_DELTA_C}°C)",
            out2,
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
