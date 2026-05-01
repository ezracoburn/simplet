import os
import re
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

OUTPUT_ROOT = "/Users/ezracoburn/Documents/Simple/output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# How many frames to sample per flight
NUM_FRAMES = 30

# Texture window for local std
TEX_WIN = 9

# Histogram bins
BINS = 256

# Robust range for binning texture values (percentiles within each frame)
P_LO = 1
P_HI = 99

# Plotting
LINE_ALPHA = 0.25
PLOT_DPI = 180


def sanitize_name(s: str) -> str:
    s = s.strip().replace(os.sep, "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180]


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

        # only canonical originals
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
    return raw.astype(np.float32) * 0.1 - 273.15


def local_std_texture(temp_c: np.ndarray, finite: np.ndarray, win: int = TEX_WIN):
    filled = temp_c.copy()
    fill_val = float(np.nanmedian(temp_c[finite])) if np.any(finite) else 0.0
    filled[~finite] = fill_val

    mean = uniform_filter(filled, win)
    mean_sq = uniform_filter(filled**2, win)
    var = np.maximum(mean_sq - mean**2, 0)
    return np.sqrt(var)


def load_texture_values(tiff_path: str):
    with rasterio.open(tiff_path) as src:
        raw = src.read(1)

    temp_c = to_celsius_autel(raw)
    temp_c[(temp_c < -50) | (temp_c > 200)] = np.nan
    finite = np.isfinite(temp_c)
    if not np.any(finite):
        return None

    tex = local_std_texture(temp_c, finite, win=TEX_WIN)
    vals = tex[finite]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return vals


def overlay_histograms_for_flight(flight_root: str):
    tiffs, media_dirs = list_camera_tiffs(flight_root)
    if not tiffs:
        print(f"\nNo camera TIFFs found under: {flight_root}")
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
        return

    selected = pick_evenly_spaced(tiffs, NUM_FRAMES)
    print(f"\nFlight: {flight_root}")
    print(f"Selected {len(selected)} frames (evenly spaced).")

    tex_vals_list = []
    frame_labels = []
    frame_los = []
    frame_his = []

    for p in selected:
        vals = load_texture_values(p)
        if vals is None:
            continue
        tex_vals_list.append(vals)
        frame_labels.append(os.path.basename(p))
        frame_los.append(np.percentile(vals, P_LO))
        frame_his.append(np.percentile(vals, P_HI))

    if len(tex_vals_list) < 5:
        print("Too few usable frames for histogram overlay (need >= 5).")
        return

    # Global bin range from per-frame robust ranges
    lo = float(np.min(frame_los))
    hi = float(np.max(frame_his))
    if hi <= lo:
        print("Degenerate texture range; skipping.")
        return

    bin_edges = np.linspace(lo, hi, BINS + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    H = []
    for vals in tex_vals_list:
        hist, _ = np.histogram(vals, bins=bin_edges)
        hist = hist.astype(np.float64)
        s = hist.sum()
        if s > 0:
            hist /= s  # normalize to probability mass for comparability
        H.append(hist)

    H = np.vstack(H)  # (n_frames, BINS)
    mean_hist = H.mean(axis=0)
    med_hist = np.median(H, axis=0)

    # Plot overlay
    plt.figure(figsize=(9, 5))
    for i in range(H.shape[0]):
        plt.plot(centers, H[i], alpha=LINE_ALPHA)


    flight_name = sanitize_name(os.path.basename(flight_root))
    plt.title(f"Texture histogram overlay (local std win={TEX_WIN})\n{flight_name}  (n={H.shape[0]} frames, normalized)")
    plt.xlabel("Texture (local std of °C)")
    plt.ylabel("Normalized count (probability mass)")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_ROOT, f"{flight_name}_texture_hist_overlay_{H.shape[0]}frames.png")
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()

    # Also save the matrix for later analysis
    out_npz = os.path.join(OUTPUT_ROOT, f"{flight_name}_texture_hist_overlay_{H.shape[0]}frames.npz")
    np.savez_compressed(out_npz, centers=centers, H=H, mean=mean_hist, median=med_hist, flight_root=flight_root)

    print("Saved:")
    print(" ", out_path)
    print(" ", out_npz)


def main():
    # Edit this list to run multiple flights in one go.
    FLIGHT_ROOTS = [
        '/Volumes/EXTERNAL HD/Thermal Flights/6 July 23',
    ]

    for fr in FLIGHT_ROOTS:
        overlay_histograms_for_flight(fr)


if __name__ == "__main__":
    main()
