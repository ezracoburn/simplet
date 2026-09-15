import os
import numpy as np
from PIL import Image
import rasterio

OUTPUT_ROOT = "/Users/ezracoburn/Documents/Simple/output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

def normalize_to_uint8(a: np.ndarray, p_lo=2.0, p_hi=98.0) -> np.ndarray:
    a = a.astype(np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise RuntimeError("No finite values in TIFF array.")
    lo = np.percentile(a, p_lo)
    hi = np.percentile(a, p_hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi

def tiff_to_uint8(img: np.ndarray, p_lo=2.0, p_hi=98.0) -> np.ndarray:
    img = img.astype(np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        raise RuntimeError("No finite values in TIFF.")
    lo = np.percentile(img[finite], p_lo)
    hi = np.percentile(img[finite], p_hi)
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)

def edge_mask_from_uint8(gray: np.ndarray, thresh_percentile=97.5) -> np.ndarray:
    # Simple gradient magnitude edge detector (no extra deps)
    g = gray.astype(np.float32) / 255.0
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    mag = np.sqrt(gx * gx + gy * gy)

    t = np.percentile(mag, thresh_percentile)
    edges = mag >= t
    return edges

def overlay_edges_on_rgb(rgb: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    # draw edges in red
    out[edges, 0] = 255
    out[edges, 1] = 0
    out[edges, 2] = 0
    return out

def main(jpg_path: str, tiff_path: str):
    # Load JPG
    jpg = Image.open(jpg_path).convert("RGB")
    jpg_np = np.array(jpg)

    # Load TIFF band 1
    with rasterio.open(tiff_path) as src:
        tif = src.read(1)

    # Convert TIFF to visible grayscale
    tif_u8 = tiff_to_uint8(tif)

    # If dimensions differ, this is already a big clue.
    print("JPG size:", (jpg_np.shape[1], jpg_np.shape[0]))
    print("TIFF size:", (tif_u8.shape[1], tif_u8.shape[0]))

    # If sizes differ, we will center-pad/crop TIFF to match JPG for visualization ONLY.
    # (We are not claiming geometric correctness here; this is just to reveal whether TIFF content covers the whole JPG.)
    if tif_u8.shape[:2] != jpg_np.shape[:2]:
        H, W = jpg_np.shape[0], jpg_np.shape[1]
        h, w = tif_u8.shape[0], tif_u8.shape[1]
        out = np.zeros((H, W), dtype=np.uint8)

        # place TIFF centered into output canvas
        y0 = max((H - h) // 2, 0)
        x0 = max((W - w) // 2, 0)
        y1 = min(y0 + h, H)
        x1 = min(x0 + w, W)

        ty0 = max((h - H) // 2, 0)
        tx0 = max((w - W) // 2, 0)
        ty1 = ty0 + (y1 - y0)
        tx1 = tx0 + (x1 - x0)

        out[y0:y1, x0:x1] = tif_u8[ty0:ty1, tx0:tx1]
        tif_u8 = out
        print("NOTE: TIFF and JPG sizes differed; TIFF was centered into JPG canvas for visualization.")

    # Save normalized TIFF view
    Image.fromarray(tif_u8, mode="L").save(os.path.join(OUTPUT_ROOT, "tiff_normalized.png"))

    # Edges from TIFF, overlay on JPG
    edges = edge_mask_from_uint8(tif_u8, thresh_percentile=97.5)
    over = overlay_edges_on_rgb(jpg_np, edges)
    Image.fromarray(over, mode="RGB").save(os.path.join(OUTPUT_ROOT, "jpg_with_tiff_edges.png"))

    print("Wrote:")
    print(" ", os.path.join(OUTPUT_ROOT, "tiff_normalized.png"))
    print(" ", os.path.join(OUTPUT_ROOT, "jpg_with_tiff_edges.png"))

if __name__ == "__main__":
    # Change these two paths to a matching pair in one folder:
    JPG = r'/Volumes/EXTERNAL HD/Thermal Flights/1 July 23/Kikirahamea - Hiva Hiva/104MEDIA/IRX_1020.jpg'
    TIF = r'/Volumes/EXTERNAL HD/Thermal Flights/1 July 23/Kikirahamea - Hiva Hiva/104MEDIA/IRX_1020.TIFF'
    main(JPG, TIF)

