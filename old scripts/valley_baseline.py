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

# Guard: require at least this fraction of pixels to be "smooth candidates"
MIN_SMOOTH_FRAC = 0.10

# Within smooth candidates, keep only the warmer subset (percentile in [50,70] is typical)
WARM_SUBSET_PERCENTILE = 60

# Baseline definition: p90 of temps within warm smooth subset
BASELINE_PERCENTILE = 90

# Histogram bins for texture thresholding
TEX_BINS = 256


def find_frame_id(path: str):
    m = re.search(r"IRX_(\d{4})", os.path.basename(path))
    return m.group(1) if m else None


def list_camera_tiffs(flight_root: str):
    tiffs = (
        glob.glob(os.path.join(flight_root, "**", "IRX_*.tif"), recursive=True)
        + glob.glob(os.path.join(flight_root, "**", "IRX_*.tiff"), recursive=True)
        + glob.glob(os.path.join(flight_root, "**", "IRX_*.TIF"), recursive=True)
        + glob.glob(os.path.join(flight_root, "**", "IRX_*.TIFF"), recursive=True)
    )

    out = []
    for p in sorted(set(tiffs)):
        name = os.path.basename(p)

        # Keep only originals like IRX_0001.TIFF (exclude suffixes like _tiff, _FLIR, etc.)
        if not re.fullmatch(r"IRX_\d{4}\.(tif|tiff|TIF|TIFF)", name):
            continue

        fid = find_frame_id(p)
        if fid is None:
            continue

        base = name.lower()
        if "rpeg" in base or "rgb" in base or "mosaic" in base or "orth" in base:
            continue

        out.append(p)
    return out


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


def smooth_1d(x: np.ndarray, k: int = 7) -> np.ndarray:
    if k <= 1:
        return x
    k = int(k)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(xp, kernel, mode="valid")


def valley_after_peak_threshold(hist: np.ndarray, bin_edges: np.ndarray, smooth_k: int = 9) -> float:
    """
    Find the "valley after the main peak" in a histogram.
    1) Smooth histogram (moving average).
    2) Find peak index (argmax).
    3) Find first local minimum after peak (h[i-1] > h[i] <= h[i+1]).
       If none found, fall back to a drop-to-fraction rule.
    Returns threshold value in texture units (bin center).
    """
    hist = hist.astype(np.float64)
    hs = smooth_1d(hist, smooth_k)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    peak_idx = int(np.argmax(hs))
    for i in range(peak_idx + 1, len(hs) - 1):
        if hs[i - 1] > hs[i] and hs[i] <= hs[i + 1]:
            return float(centers[i])

    peak_h = hs[peak_idx]
    cutoff = peak_h * 0.05
    for i in range(peak_idx + 1, len(hs)):
        if hs[i] <= cutoff:
            return float(centers[i])

    return float(centers[min(peak_idx + 1, len(centers) - 1)])


def texture_hist_and_threshold(tex_vals: np.ndarray, nbins: int = TEX_BINS):
    tex_vals = tex_vals[np.isfinite(tex_vals)]
    if tex_vals.size == 0:
        return None, None, None

    lo, hi = np.percentile(tex_vals, 1), np.percentile(tex_vals, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(tex_vals))
        hi = float(np.max(tex_vals))
        if hi <= lo:
            return None, None, None

    hist, bin_edges = np.histogram(tex_vals, bins=nbins, range=(lo, hi))
    thr = valley_after_peak_threshold(hist, bin_edges)
    return hist, bin_edges, thr


def save_texture_hist_viz(frame_id: str, base: str, hist, bin_edges, thr: float):
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    plt.figure(figsize=(8, 4))
    plt.plot(centers, hist)
    plt.axvline(thr, linewidth=2)
    plt.title(f"{base}\nTexture hist (local std); valley-threshold={thr:.6f}")
    plt.xlabel("Texture (local std of °C)")
    plt.ylabel("Count")
    plt.tight_layout()
    out_hist = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_texture_hist_valley.png")
    plt.savefig(out_hist, dpi=160)
    plt.close()


def process_one(flight_root: str, tiff_path: str):
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

    hist, bin_edges, thr = texture_hist_and_threshold(tex_vals, nbins=TEX_BINS)
    if hist is None:
        print(f"{base}: SKIP (texture hist failed)")
        return

    S = finite & (tex <= thr)
    smooth_frac = float(np.count_nonzero(S)) / float(np.count_nonzero(finite))

    if frame_id:
        save_texture_hist_viz(frame_id, base, hist, bin_edges, thr)

    print(base)
    print(f"  raw_min={raw_min}  raw_max={raw_max}")
    print(f"  temp_min={min_c:.2f}°C  temp_max={max_c:.2f}°C")
    print(f"  valley_thr(texture)={thr:.6f}  smooth_frac={smooth_frac:.4f}")

    baseline_c = None
    warm_thr = None

    if smooth_frac < MIN_SMOOTH_FRAC:
        print(f"  BASELINE SKIP (smooth_frac < {MIN_SMOOTH_FRAC:.2f})\n")
    else:
        smooth_temps = temp_c[S]
        warm_thr = float(np.nanpercentile(smooth_temps, WARM_SUBSET_PERCENTILE))
        S2 = S & (temp_c >= warm_thr)

        if np.count_nonzero(S2) < 100:
            print(f"  BASELINE SKIP (warm smooth subset too small)\n")
        else:
            baseline_c = float(np.nanpercentile(temp_c[S2], BASELINE_PERCENTILE))
            print(f"  warm_thr = p{WARM_SUBSET_PERCENTILE}(S) = {warm_thr:.3f} °C")
            print(f"  BASELINE ≈ p{BASELINE_PERCENTILE}(S2) = {baseline_c:.3f} °C\n")

    disp_gray = normalize_for_display(temp_c)

    out1 = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_S_smooth_on_gray.png")
    overlay_and_save(
        disp_gray,
        S,
        f"{base}\nS: smooth (low texture) frac={smooth_frac:.3f}",
        out1
    )

    if warm_thr is not None:
        S2 = S & (temp_c >= warm_thr)
        out2 = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_S2_warm_smooth_on_gray.png")
        overlay_and_save(
            disp_gray,
            S2,
            f"{base}\nS2: warm smooth (>=p{WARM_SUBSET_PERCENTILE}(S))",
            out2
        )

    if baseline_c is not None:
        below = finite & (temp_c <= baseline_c)
        out3 = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_below_baseline_p{BASELINE_PERCENTILE}_on_gray.png")
        overlay_and_save(
            disp_gray,
            below,
            f"{base}\nBelow baseline p{BASELINE_PERCENTILE}={baseline_c:.2f}°C",
            out3
        )


def main(flight_root: str):
    tiffs = list_camera_tiffs(flight_root)
    if not tiffs:
        raise SystemExit(f"No camera TIFFs found under: {flight_root} (expected IRX_####.tif/.TIFF)")

    selected = pick_evenly_spaced(tiffs, NUM_IMAGES)
    print("Selected frames:")
    for p in selected:
        print(" ", os.path.basename(p))
    print("")

    for p in selected:
        process_one(flight_root, p)


if __name__ == "__main__":
    FLIGHT_ROOT = "/Volumes/EXTERNAL HD/Thermal Flights/1 July 23/Kikirahamea - Hiva Hiva"
    main(FLIGHT_ROOT)
