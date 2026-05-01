import os
import re
import glob
import json
import math
import numpy as np
import rasterio

from scipy.fft import fft2, ifft2
from pyproj import CRS, Transformer

from georef_autel_640t import read_frame_meta_from_jpg, THERMAL_WIDTH_PX, THERMAL_HEIGHT_PX

OUTPUT_ROOT = "/Users/ezracoburn/Documents/Simple/output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

REPORT_PATH = os.path.join(OUTPUT_ROOT, "gsd_calibration_report.txt")
RESULT_JSON = os.path.join(OUTPUT_ROOT, "gsd_calibration_result.json")


def utm_crs(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_dict({"proj": "utm", "zone": zone, "south": lat < 0})


def to_utm_xy(lon: float, lat: float):
    crs = utm_crs(lon, lat)
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = t.transform(lon, lat)
    return crs, float(x), float(y)


def robust_norm(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    finite = np.isfinite(a)
    if not np.any(finite):
        raise RuntimeError("No finite values in TIFF.")
    lo = np.percentile(a[finite], 2)
    hi = np.percentile(a[finite], 98)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def phase_corr_shift_and_quality(im1: np.ndarray, im2: np.ndarray):
    """
    Returns dy, dx (pixels) and a quality score (peak/median_abs).
    """
    F1 = fft2(im1)
    F2 = fft2(im2)
    R = F1 * np.conj(F2)
    R /= np.maximum(np.abs(R), 1e-12)
    r = np.real(ifft2(R))

    H, W = r.shape
    maxpos = np.unravel_index(np.argmax(r), r.shape)
    peak = float(r[maxpos])

    dy, dx = float(maxpos[0]), float(maxpos[1])
    if dy > H / 2:
        dy -= H
    if dx > W / 2:
        dx -= W

    med = float(np.median(np.abs(r)))
    quality = peak / (med + 1e-12)
    return dy, dx, quality


def parse_frame_number(path: str):
    base = os.path.basename(path)
    m = re.search(r"_(\d+)\.", base)
    return int(m.group(1)) if m else None


def find_pairs(flight_root: str):
    jpgs = glob.glob(os.path.join(flight_root, "**", "*.jpg"), recursive=True) + \
           glob.glob(os.path.join(flight_root, "**", "*.JPG"), recursive=True)

    tifs = glob.glob(os.path.join(flight_root, "**", "*.tif"), recursive=True) + \
           glob.glob(os.path.join(flight_root, "**", "*.tiff"), recursive=True) + \
           glob.glob(os.path.join(flight_root, "**", "*.TIF"), recursive=True) + \
           glob.glob(os.path.join(flight_root, "**", "*.TIFF"), recursive=True)

    jpg_by_n = {}
    for p in jpgs:
        n = parse_frame_number(p)
        if n is not None:
            jpg_by_n[n] = p

    tif_by_n = {}
    for p in tifs:
        n = parse_frame_number(p)
        if n is not None:
            tif_by_n[n] = p

    ns = sorted(set(jpg_by_n.keys()) & set(tif_by_n.keys()))
    return [{"n": n, "jpg": jpg_by_n[n], "tif": tif_by_n[n]} for n in ns]


def calibrate_scalar_gsd(
    flight_root: str,
    max_pairs: int = 400,
    min_pix_shift: float = 10.0,
    min_quality: float = 10.0
):
    pairs = find_pairs(flight_root)
    if len(pairs) < 2:
        raise SystemExit("Not enough jpg+tif pairs found.")

    consec = []
    for i in range(len(pairs) - 1):
        if pairs[i + 1]["n"] == pairs[i]["n"] + 1:
            consec.append((pairs[i], pairs[i + 1]))
    if not consec:
        raise SystemExit("No consecutive pairs found.")

    consec = consec[:max_pairs]

    gsd_samples = []
    qual_samples = []
    alt_samples = []
    used = 0
    rejected = 0

    log_lines = []
    log_lines.append(f"Flight root: {flight_root}")
    log_lines.append(f"Consecutive pairs considered: {len(consec)}")
    log_lines.append(f"Filters: min_pix_shift={min_pix_shift}, min_quality={min_quality}")
    log_lines.append("")

    for a, b in consec:
        meta_a = read_frame_meta_from_jpg(a["jpg"])
        meta_b = read_frame_meta_from_jpg(b["jpg"])

        _, xa, ya = to_utm_xy(meta_a.lon, meta_a.lat)
        _, xb, yb = to_utm_xy(meta_b.lon, meta_b.lat)

        ground_shift = math.hypot(xb - xa, yb - ya)

        with rasterio.open(a["tif"]) as src:
            t1 = src.read(1)
        with rasterio.open(b["tif"]) as src:
            t2 = src.read(1)
        if t1.shape != t2.shape:
            rejected += 1
            continue

        im1 = robust_norm(t1)
        im2 = robust_norm(t2)

        dy, dx, q = phase_corr_shift_and_quality(im1, im2)
        pix_shift = math.hypot(dx, dy)

        if pix_shift < min_pix_shift or q < min_quality:
            rejected += 1
            continue

        gsd = ground_shift / pix_shift
        gsd_samples.append(gsd)
        qual_samples.append(q)
        alt_samples.append((meta_a.alt_agl_m + meta_b.alt_agl_m) / 2.0)
        used += 1

        log_lines.append(
            f"pair {a['n']}->{b['n']}  ground={ground_shift:7.2f} m  "
            f"pix=({dx:7.2f},{dy:7.2f}) |d|={pix_shift:7.2f}  q={q:6.2f}  gsd={gsd:0.6f}"
        )

    if used < 10:
        raise SystemExit(f"Too few usable pairs ({used}). Try lowering thresholds or using a more textured segment.")

    gsd_arr = np.array(gsd_samples, dtype=np.float64)

    gsd_med = float(np.median(gsd_arr))
    mad = float(np.median(np.abs(gsd_arr - gsd_med)))
    p10 = float(np.percentile(gsd_arr, 10))
    p90 = float(np.percentile(gsd_arr, 90))

    alt_med = float(np.median(np.array(alt_samples, dtype=np.float64)))
    q_med = float(np.median(np.array(qual_samples, dtype=np.float64)))

    # Implied effective FOV at median altitude
    ground_w = gsd_med * THERMAL_WIDTH_PX
    ground_h = gsd_med * THERMAL_HEIGHT_PX
    hfov_eff = float(2.0 * math.degrees(math.atan((ground_w / 2.0) / alt_med)))
    vfov_eff = float(2.0 * math.degrees(math.atan((ground_h / 2.0) / alt_med)))

    log_lines.append("")
    log_lines.append("==== SUMMARY (robust) ====")
    log_lines.append(f"usable_pairs: {used}   rejected_pairs: {rejected}")
    log_lines.append(f"median_quality: {q_med:.2f}")
    log_lines.append(f"median_alt_agl_m: {alt_med:.3f}")
    log_lines.append(f"median_gsd_m_per_px: {gsd_med:.6f}")
    log_lines.append(f"mad_gsd_m_per_px: {mad:.6f}")
    log_lines.append(f"p10_gsd_m_per_px: {p10:.6f}")
    log_lines.append(f"p90_gsd_m_per_px: {p90:.6f}")
    log_lines.append(f"implied_ground_width_m: {ground_w:.3f}")
    log_lines.append(f"implied_ground_height_m: {ground_h:.3f}")
    log_lines.append(f"implied_effective_hfov_deg: {hfov_eff:.3f}")
    log_lines.append(f"implied_effective_vfov_deg: {vfov_eff:.3f}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    out = {
        "flight_root": flight_root,
        "usable_pairs": used,
        "rejected_pairs": rejected,
        "median_quality": q_med,
        "median_alt_agl_m": alt_med,
        "median_gsd_m_per_px": gsd_med,
        "mad_gsd_m_per_px": mad,
        "p10_gsd_m_per_px": p10,
        "p90_gsd_m_per_px": p90,
        "implied_ground_width_m": ground_w,
        "implied_ground_height_m": ground_h,
        "implied_effective_hfov_deg": hfov_eff,
        "implied_effective_vfov_deg": vfov_eff,
        "thermal_width_px": THERMAL_WIDTH_PX,
        "thermal_height_px": THERMAL_HEIGHT_PX,
        "notes": "Scalar GSD estimated from |UTM delta| / |pixel shift| over consecutive pairs; robust stats reported as uncertainty."
    }
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote report: {REPORT_PATH}")
    print(f"Wrote result: {RESULT_JSON}")
    print("")
    print("Recommended for footprints:")
    print(f"  GSD_M_PER_PX = {gsd_med:.6f}  (MAD {mad:.6f}, p10 {p10:.6f}, p90 {p90:.6f})")
    print("Implied effective FOV at median altitude:")
    print(f"  HFOV_EFF_DEG = {hfov_eff:.3f}")
    print(f"  VFOV_EFF_DEG = {vfov_eff:.3f}")


if __name__ == "__main__":
    FLIGHT_ROOT = r'/Volumes/EXTERNAL HD/Thermal Flights/1 July 23/Kikirahamea - Hiva Hiva'
    calibrate_scalar_gsd(
        flight_root=FLIGHT_ROOT,
        max_pairs=400,
        min_pix_shift=10.0,
        min_quality=10.0
    )
