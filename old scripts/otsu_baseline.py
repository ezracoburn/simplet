import os
import re
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter

OUTPUT_ROOT = "/Users/ezracoburn/Documents/Simple/output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

NUM_IMAGES = 5
MIN_OCEAN_FRAC = 0.10
TEMP_PERCENTILE = 90
OTSU_BINS = 256


def find_frame_id(path: str):
    m = re.search(r"IRX_(\d{4})", os.path.basename(path))
    return m.group(1) if m else None


def list_camera_tiffs(flight_root: str):
    tiffs = glob.glob(os.path.join(flight_root, "**", "IRX_*.tif"), recursive=True) + \
            glob.glob(os.path.join(flight_root, "**", "IRX_*.tiff"), recursive=True) + \
            glob.glob(os.path.join(flight_root, "**", "IRX_*.TIF"), recursive=True) + \
            glob.glob(os.path.join(flight_root, "**", "IRX_*.TIFF"), recursive=True)

    out = []
    for p in sorted(set(tiffs)):
        name = os.path.basename(p)

        # keep only originals like IRX_0001.TIFF (no suffixes like _tiff, _FLIR, etc.)
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


def load_matching_jpg(flight_root: str, frame_id: str):
    candidates = glob.glob(os.path.join(flight_root, "**", f"IRX_{frame_id}.jpg"), recursive=True) + \
                 glob.glob(os.path.join(flight_root, "**", f"IRX_{frame_id}.JPG"), recursive=True)
    return candidates[0] if candidates else None


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


def overlay_and_save(background, mask, title, out_path):
    plt.figure(figsize=(8, 6))

    # Force true grayscale for 2D images
    if background.ndim == 2:
        plt.imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
    else:
        plt.imshow(background)

    # Draw mask as pure red with alpha (no rainbow colormap)
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = 1.0  # R
    overlay[..., 3] = mask.astype(np.float32) * 0.45  # alpha only where mask is True
    plt.imshow(overlay)

    plt.title(title)
    plt.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


''' def grad_texture(temp_c: np.ndarray, finite: np.ndarray) -> np.ndarray:
    # Replace invalid pixels so gradients don't become NaN everywhere
    filled = temp_c.copy()
    if np.any(finite):
        fill_val = float(np.nanmedian(temp_c[finite]))
    else:
        fill_val = 0.0
    filled[~finite] = fill_val

    dx = np.abs(np.diff(filled, axis=1, prepend=filled[:, :1]))
    dy = np.abs(np.diff(filled, axis=0, prepend=filled[:1, :]))
    return dx + dy '''


def local_std_texture(temp_c: np.ndarray, finite: np.ndarray, win: int = 9):
    filled = temp_c.copy()
    fill_val = np.nanmedian(temp_c[finite])
    filled[~finite] = fill_val

    mean = uniform_filter(filled, win)
    mean_sq = uniform_filter(filled**2, win)
    var = np.maximum(mean_sq - mean**2, 0)
    return np.sqrt(var)


def otsu_threshold(x: np.ndarray, nbins: int = OTSU_BINS):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, None, None, None

    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi <= lo:
            return np.nan, None, None, None

    hist, bin_edges = np.histogram(x, bins=nbins, range=(lo, hi))
    hist = hist.astype(np.float64)
    p = hist / (hist.sum() + 1e-12)

    omega = np.cumsum(p)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]

    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    k = int(np.nanargmax(sigma_b2))
    thr = float(centers[k])
    return thr, hist, bin_edges, sigma_b2


def save_otsu_viz(frame_id: str, base: str, tex_vals: np.ndarray, thr: float, hist, bin_edges, sigma_b2):
    # 1) histogram + threshold line
    plt.figure(figsize=(8, 4))
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    plt.plot(centers, hist)
    plt.axvline(thr, linewidth=2)
    plt.title(f"{base}\nOtsu threshold on texture: thr={thr:.4f}")
    plt.xlabel("Texture (|dT/dx|+|dT/dy|)")
    plt.ylabel("Count")
    plt.tight_layout()
    out_hist = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_otsu_texture_hist.png")
    plt.savefig(out_hist, dpi=160)
    plt.close()

    # 2) between-class variance curve (optional but useful)
    plt.figure(figsize=(8, 4))
    plt.plot(centers, sigma_b2)
    plt.axvline(thr, linewidth=2)
    plt.title(f"{base}\nBetween-class variance (Otsu)")
    plt.xlabel("Texture threshold")
    plt.ylabel("Between-class variance")
    plt.tight_layout()
    out_var = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_otsu_between_class_variance.png")
    plt.savefig(out_var, dpi=160)
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

    # Raw/C diagnostics (as requested)
    raw_valid = raw[finite]
    raw_valid = raw_valid[raw_valid > 0]
    if raw_valid.size == 0:
        print(f"{base}: SKIP (no valid raw temps > 0)")
        return

    raw_min = int(raw_valid.min())
    raw_max = int(raw_valid.max())
    min_c = raw_min * 0.1 - 273.15
    max_c = raw_max * 0.1 - 273.15

    # Texture + Otsu
    tex = local_std_texture(temp_c, finite)
    tex_vals = tex[finite]
    thr, hist, bin_edges, sigma_b2 = otsu_threshold(tex_vals, nbins=OTSU_BINS)

    if not np.isfinite(thr) or hist is None:
        print(f"{base}: SKIP (otsu failed)")
        return

    cand = finite & (tex <= thr)
    cand_frac = float(np.count_nonzero(cand)) / float(np.count_nonzero(finite))

    # Always save Otsu visualization (even if we skip baseline)
    if frame_id:
        save_otsu_viz(frame_id, base, tex_vals, thr, hist, bin_edges, sigma_b2)

    print(base)
    print(f"  raw_min={raw_min}  raw_max={raw_max}")
    print(f"  temp_min={min_c:.2f}°C  temp_max={max_c:.2f}°C")
    print(f"  otsu_thr(texture)={thr:.6f}  cand_frac={cand_frac:.4f}")

    if cand_frac < MIN_OCEAN_FRAC:
        print(f"  BASELINE SKIP (cand_frac < {MIN_OCEAN_FRAC:.2f})\n")
        # still save overlays so you can see why it failed
    else:
        baseline_c = float(np.nanpercentile(temp_c[cand], TEMP_PERCENTILE))
        print(f"  BASELINE ≈ p{TEMP_PERCENTILE}(cand) = {baseline_c:.3f} °C\n")

    # Overlays: candidate mask on grayscale + thermal jpg (like your mode script)
    disp_gray = normalize_for_display(temp_c)
    out1 = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_otsu_cand_on_gray.png")
    overlay_and_save(
        disp_gray,
        cand,
        f"{base}\nOtsu cand (low-texture) frac={cand_frac:.3f}",
        out1
    )

    jpg_path = load_matching_jpg(flight_root, frame_id) if frame_id else None
    if jpg_path:
        jpg = Image.open(jpg_path).convert("RGB")
        jpg_arr = np.asarray(jpg)
        if jpg_arr.shape[0] == cand.shape[0] and jpg_arr.shape[1] == cand.shape[1]:
            out2 = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_otsu_cand_on_jpg.png")
            overlay_and_save(
                jpg_arr,
                cand,
                f"{os.path.basename(jpg_path)}\nOtsu cand (low-texture) frac={cand_frac:.3f}",
                out2
            )
        else:
            print(f"  JPG overlay skipped (size mismatch: jpg={jpg_arr.shape[:2]} tif={cand.shape})")
    else:
        print("  JPG overlay skipped (no matching JPG found)")


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
