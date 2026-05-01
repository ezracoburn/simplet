import os
import glob
import math
from typing import Tuple, List

import simplekml
from georef_autel_640t import (
    read_frame_meta_from_jpg,
    thermal_pixel_to_lonlat,
    THERMAL_HFOV_DEG,
    THERMAL_VFOV_DEG,
    THERMAL_WIDTH_PX,
    THERMAL_HEIGHT_PX,
)

# =========================
# OUTPUT LOCATION (FIXED)
# =========================

OUTPUT_ROOT = "/Users/ezracoburn/Documents/Simple/output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def footprint_dims_m(alt_agl_m: float) -> Tuple[float, float]:
    hfov = math.radians(THERMAL_HFOV_DEG)
    vfov = math.radians(THERMAL_VFOV_DEG)
    w = 2.0 * alt_agl_m * math.tan(hfov / 2.0)
    h = 2.0 * alt_agl_m * math.tan(vfov / 2.0)
    return w, h


def corners_lonlat(meta) -> List[Tuple[float, float]]:
    pts = [
        (0, 0),
        (THERMAL_WIDTH_PX - 1, 0),
        (THERMAL_WIDTH_PX - 1, THERMAL_HEIGHT_PX - 1),
        (0, THERMAL_HEIGHT_PX - 1),
    ]
    out = []
    for x, y in pts:
        lon, lat = thermal_pixel_to_lonlat(x, y, meta)
        out.append((lon, lat))
    out.append(out[0])
    return out


def main(folder: str, max_items: int = 10, write_kml: bool = True):
    jpgs = sorted(glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True) +
                  glob.glob(os.path.join(folder, "**", "*.JPG"), recursive=True))

    if not jpgs:
        raise SystemExit(f"No JPGs found under: {folder}")

    print(f"Found {len(jpgs)} JPGs. Showing first {min(max_items, len(jpgs))}.\n")

    kml = simplekml.Kml() if write_kml else None

    for i, jpg in enumerate(jpgs[:max_items], start=1):
        base = os.path.basename(jpg)
        meta = read_frame_meta_from_jpg(jpg)

        w_m, h_m = footprint_dims_m(meta.alt_agl_m)
        gsdx = w_m / THERMAL_WIDTH_PX
        gsdy = h_m / THERMAL_HEIGHT_PX

        poly = corners_lonlat(meta)

        print(f"[{i}] {base}")
        print(f"  lat/lon: {meta.lat:.8f}, {meta.lon:.8f}")
        print(f"  yaw_deg: {meta.yaw_deg:.3f}")
        print(f"  alt_agl_m: {meta.alt_agl_m:.3f}   (alt_msl_m: {meta.alt_msl_m})")
        print(f"  footprint (m): W={w_m:.3f}  H={h_m:.3f}")
        print(f"  GSD (m/px):   X={gsdx:.4f}  Y={gsdy:.4f}")
        print("  corners (lon,lat):")
        labels = ["NW", "NE", "SE", "SW", "NW(close)"]
        for lab, (lon, lat) in zip(labels, poly):
            print(f"    {lab}: {lon:.8f}, {lat:.8f}")
        print()

        if kml is not None:
            p = kml.newpolygon(name=base)
            p.outerboundaryis = poly
            p.style.polystyle.fill = 0
            p.style.linestyle.width = 2

    if kml is not None:
        out_kml = os.path.join(OUTPUT_ROOT, "sanity_footprints.kml")
        kml.save(out_kml)
        print(f"Wrote KML to: {out_kml}")


if __name__ == "__main__":
    # Change this to your flight root (the folder containing 100MEDIA/101MEDIA/etc.)
    FLIGHT_ROOT = r'/Volumes/EXTERNAL HD/Thermal Flights/1 July 23/Kikirahamea - Hiva Hiva'
    main(FLIGHT_ROOT, max_items=10, write_kml=True)

