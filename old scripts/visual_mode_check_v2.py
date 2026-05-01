import os
import re
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image

OUTPUT_ROOT = "/Users/ezracoburn/Documents/Simple/output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

NUM_IMAGES = 5


def find_frame_id(path: str):
    m = re.search(r"IRX_(\d{4})", os.path.basename(path))
    return m.group(1) if m else None


def list_camera_tiffs(flight_root: str):
    pats = [
        os.path.join(flight_root, "**", "IRX_*.tif"),
        os.path.join(flight_root, "**", "IRX_*.tiff"),
        os.path.join(flight_root, "**", "IRX_*.TIF"),
        os.path.join(flight_root, "**", "IRX_*.TIFF"),
    ]
    files = []
    for pat in pats:
        files.extend(glob.glob(pat, recursive=True))

    out = []
    for p in sorted(set(files)):
        name = os.path.basename(p)

        # KEEP ONLY exact originals like IRX_0001.TIFF (no suffixes)
        if not re.fullmatch(r"IRX_\d{4}\.(tif|tiff|TIF|TIFF)", name):
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
    plt.imshow(background)
    plt.imshow(mask, cmap="Reds", alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def exact_mode_raw_k10(values_raw: np.ndarray) -> int:
    a = values_raw
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0
    a = a.astype(np.int32)

    vals, counts = np.unique(a, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def process_one(flight_root: str, tiff_path: str):
    frame_id = find_frame_id(tiff_path)
    base = os.path.basename(tiff_path)

    with rasterio.open(tiff_path) as src:
        raw = src.read(1)
        if raw.dtype.kind not in ("u", "i", "f"):
            print(f"{base}: SKIP (unexpected dtype {raw.dtype})")
            return

    temp_c = to_celsius_autel(raw)

    # Remove obviously invalid values
    temp_c[(temp_c < -50) | (temp_c > 200)] = np.nan
    finite = np.isfinite(temp_c)
    if not np.any(finite):
        print(f"{base}: SKIP (no finite temps)")
        return

    # --- NEW: raw and Celsius range diagnostics ---
    raw_valid = raw[finite]
    raw_valid = raw_valid[raw_valid > 0]

    if raw_valid.size == 0:
        print(f"{base}: SKIP (no valid raw temps > 0)")
        return

    raw_min = int(raw_valid.min())
    raw_max = int(raw_valid.max())

    min_c = raw_min * 0.1 - 273.15
    max_c = raw_max * 0.1 - 273.15

    looks_radiometric = (2000 <= raw_min <= 5000) and (2000 <= raw_max <= 7000)

    print(base)
    print(f"  raw_min={raw_min}  raw_max={raw_max}  looks_radiometric={looks_radiometric}")
    print(f"  temp_min={min_c:.2f}°C  temp_max={max_c:.2f}°C")


    raw_valid = raw[finite]
    raw_valid = raw_valid[raw_valid > 0]
    if raw_valid.size == 0:
        print(f"{base}: SKIP (no valid raw temps > 0)")
        return

    mode_raw = exact_mode_raw_k10(raw_valid)
    mode_c = mode_raw * 0.1 - 273.15

    mode_mask = finite & (raw.astype(np.int32) == int(mode_raw))

    total_count = int(np.count_nonzero(finite))
    mode_count = int(np.count_nonzero(mode_mask))

    print(base)
    print(f"  MODE = {mode_c:.3f} °C   raw={mode_raw}   frac={mode_count/total_count:.4f}")
    print("")

    disp_gray = normalize_for_display(temp_c)
    out1 = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_mode_on_gray.png")
    overlay_and_save(
        disp_gray,
        mode_mask,
        f"{base}\nMODE={mode_c:.2f}°C  raw={mode_raw}",
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
                f"{os.path.basename(jpg_path)}\nMODE={mode_c:.2f}°C  raw={mode_raw}",
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
    FLIGHT_ROOT = '/Volumes/EXTERNAL HD/Thermal Flights/1 July 23/Hanga Roa - Rano Kau'
    main(FLIGHT_ROOT)

