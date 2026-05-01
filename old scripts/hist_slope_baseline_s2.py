import os
import re
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, label, distance_transform_edt
from collections import Counter
import json
from datetime import datetime

OUTPUT_ROOT = "/Users/ezracoburn/Documents/Simple/output/v4"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

NUM_IMAGES = 10

MIN_SMOOTH_FRAC = 0.10
BASELINE_PERCENTILE = 90
BASELINE_BAND_DELTA_C = 0.1

TEX_BINS = 256
TEX_WIN = 8

# Angle-knee params (your tuning)
HIST_SMOOTH_K = 9
ANGLE_DEG = 1.0
ANGLE_RUN = 10

# S2 mode: keep only pixels within X pixels of the largest component
S2_MODE = "largest"      # "largest" | "edge" | "within_x"
CONNECTIVITY_8 = True

DILATE_PIXELS = 0        # X pixels: keep S pixels with distance-to-largest <= X

# Optional debug output
SAVE_DIST_HEATMAP = False
DIST_HEATMAP_CLIP = 30   # clip distances for visualization (pixels)
                         #CANNOT SET TO 0 (runtime)

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

    # distance to nearest pixel in largest component
    dist = distance_transform_edt(~L)
    keep = mask & (dist <= float(x_pixels))
    return keep, dist, L


def build_s2(mask_s: np.ndarray):
    if S2_MODE == "edge":
        return keep_edge_components(mask_s, CONNECTIVITY_8), None, None
    if S2_MODE == "largest":
        L = keep_largest_component(mask_s, CONNECTIVITY_8)
        return L, None, L
    # within_x
    return keep_within_x_of_largest(mask_s, DILATE_PIXELS, CONNECTIVITY_8)


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
    out_hist = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_texture_hist_angleknee.png")
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
    out_ang = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_texture_angle_angleknee.png")
    plt.savefig(out_ang, dpi=160)
    plt.close()


def save_dist_heatmap(frame_id: str, base: str, disp_gray: np.ndarray, dist: np.ndarray, L: np.ndarray):
    if dist is None or L is None:
        return

    d = np.clip(dist, 0, DIST_HEATMAP_CLIP).astype(np.float32)
    d_norm = d / float(DIST_HEATMAP_CLIP)

    plt.figure(figsize=(8, 6))
    plt.imshow(d_norm, cmap="magma", vmin=0.0, vmax=1.0)
    plt.title(f"{base}\nDistance-to-largest heatmap (0..{DIST_HEATMAP_CLIP}px clipped)")
    plt.axis("off")
    out1 = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_dist_to_largest_heatmap.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()

    # Heatmap over grayscale for spatial context
    plt.figure(figsize=(8, 6))
    plt.imshow(disp_gray, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imshow(d_norm, cmap="magma", alpha=0.55, vmin=0.0, vmax=1.0)

    # Outline of largest component
    outline = np.zeros((*L.shape, 4), dtype=np.float32)
    outline[..., 0] = 0.0
    outline[..., 1] = 1.0
    outline[..., 2] = 1.0
    outline[..., 3] = L.astype(np.float32) * 0.25
    plt.imshow(outline)

    plt.title(f"{base}\nDist-to-largest over thermal (magma), cyan=largest component")
    plt.axis("off")
    out2 = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_dist_to_largest_on_gray.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
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

    tex = local_std_texture(temp_c, finite, TEX_WIN)
    tex_vals = tex[finite]
    hist, bin_edges = texture_hist(tex_vals, nbins=TEX_BINS)
    if hist is None:
        print(f"{base}: SKIP (texture hist failed)")
        return

    thr, angles, hs, centers = angle_knee_threshold(hist, bin_edges)

    S = finite & (tex <= thr)
    S2, dist, L = build_s2(S)

    frac_finite = float(np.count_nonzero(finite))
    s_frac = float(np.count_nonzero(S)) / frac_finite
    s2_frac = float(np.count_nonzero(S2)) / frac_finite

    if frame_id:
        save_hist_angle_debug(frame_id, base, hist, bin_edges, thr, angles, hs, centers)

    print(base)
    print(f"  angle_knee_thr(texture)={thr:.6f}")
    print(f"  S  frac={s_frac:.4f}")
    if S2_MODE == "within_x":
        print(f"  S2 frac={s2_frac:.4f}  (mode=within_x, x={DILATE_PIXELS}px, conn={'8' if CONNECTIVITY_8 else '4'})")
    else:
        print(f"  S2 frac={s2_frac:.4f}  (mode={S2_MODE}, conn={'8' if CONNECTIVITY_8 else '4'})")

        baseline_c = None
    if s2_frac < MIN_SMOOTH_FRAC:
        print(f"  BASELINE(S2) SKIP (S2 frac < {MIN_SMOOTH_FRAC:.2f})\n")
    else:
        baseline_c = float(np.nanpercentile(temp_c[S2], BASELINE_PERCENTILE))
        print(f"  BASELINE(S2) = p{BASELINE_PERCENTILE} = {baseline_c:.3f} °C\n")

    disp_gray = normalize_for_display(temp_c)

    outS = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_S_on_gray.png")
    overlay_and_save(disp_gray, S, f"{base}\nS: tex<=thr  frac={s_frac:.3f}", outS)

    outS2 = os.path.join(OUTPUT_ROOT, f"IRX_{frame_id}_S2_on_gray.png")
    if S2_MODE == "within_x":
        title = f"{base}\nS2: within {DILATE_PIXELS}px of largest  frac={s2_frac:.3f}"
    else:
        title = f"{base}\nS2: contiguous ({S2_MODE})  frac={s2_frac:.3f}"
    overlay_and_save(disp_gray, S2, title, outS2)

    if SAVE_DIST_HEATMAP and S2_MODE == "within_x":
        save_dist_heatmap(frame_id, base, disp_gray, dist, L)

    if baseline_c is not None:
        band = S2 & (np.abs(temp_c - baseline_c) <= BASELINE_BAND_DELTA_C)
        overlay_and_save(
            disp_gray,
            band,
            f"{base}\nBaseline p{BASELINE_PERCENTILE}(S2)={baseline_c:.2f}°C (±{BASELINE_BAND_DELTA_C}°C)",
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
    FLIGHT_ROOT = '/Volumes/EXTERNAL HD/Thermal Flights/5 July 23/South to Vai Mata'
    main(FLIGHT_ROOT)
