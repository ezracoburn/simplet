import os
import re
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image

OUTPUT_ROOT = "/Users/ezracoburn/Documents/Simple/output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

NUM_BUCKETS = 50
NUM_IMAGES = 5

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
        fid = find_frame_id(p)
        if fid is None:
            continue
        base = os.path.basename(p).lower()
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

def robust_edges(values: np.ndarray, bins: int):
    lo = np.percentile(values, 1)
    hi = np.percentile(values, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi <= lo:
            hi = lo + 1.0
    return np.linspace(lo, hi, bins + 1, dtype=np.float32)

def mode_bin(values: np.ndarray, edges: np.ndarray):
    hist, _ = np.histogram(values, bins=edges)
    i = int(np.argmax(hist))
    lo = float(edges[i])
    hi = float(edges[i + 1])
    center = 0.5 * (lo + hi)
    return lo, hi, center, int(hist[i]), int(hist.sum())

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
    plt.imshow(background)
    plt.imshow(mask, cmap="Reds", alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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

    temps = temp_c[finite]
    edges = robust_edges(temps, NUM_BUCKETS)
    bin_lo, bin_hi, mode_c, mode_count, total_count = mode_bin(temps, edges)

    mode_mask = (temp_c >= bin_lo) & (temp_c < bin_hi) & finite

    print(base)
    print(f"  MODE ≈ {mode_c:.3f} °C   bin=[{bin_lo:.3f}, {bin_hi:.3f})   frac={mode_count/total_count:.4f}")
    print("")

    disp_gray = normalize_for_display(temp_c)
    out1 = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_mode_on_gray.png")
    overlay_and_save(
        disp_gray,
        mode_mask,
        f"{base}\nMODE≈{mode_c:.2f}°C  bin=[{bin_lo:.2f},{bin_hi:.2f})",
        out1
    )

    jpg_path = load_matching_jpg(flight_root, frame_id) if frame_id else None
    if jpg_path:
        jpg = Image.open(jpg_path).convert("RGB")
        jpg_arr = np.asarray(jpg)
        if jpg_arr.shape[0] == mode_mask.shape[0] and jpg_arr.shape[1] == mode_mask.shape[1]:
            out2 = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_mode_on_jpg.png")
            overlay_and_save(
                jpg_arr,
                mode_mask,
                f"{os.path.basename(jpg_path)}\nMODE≈{mode_c:.2f}°C  bin=[{bin_lo:.2f},{bin_hi:.2f})",
                out2
            )
        else:
            print(f"  JPG overlay skipped (size mismatch: jpg={jpg_arr.shape[:2]} tif={mode_mask.shape})")
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
