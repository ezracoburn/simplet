import re
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import exifread
from pyproj import CRS, Transformer


# =========================
# HARD-CODED CAMERA PARAMS
# =========================

THERMAL_HFOV_DEG = 33.0
THERMAL_VFOV_DEG = 26.0

THERMAL_WIDTH_PX = 640
THERMAL_HEIGHT_PX = 512


# =========================
# DATA MODEL
# =========================

@dataclass(frozen=True)
class FrameMeta:
    lat: float
    lon: float
    yaw_deg: float          # clockwise from north
    alt_agl_m: float        # meters above ground
    alt_msl_m: Optional[float]


# =========================
# EXIF / XMP HELPERS
# =========================

def _ratio_to_float(r) -> float:
    return float(r.num) / float(r.den)


def _dms_to_deg(dms, ref) -> float:
    deg = _ratio_to_float(dms.values[0])
    minutes = _ratio_to_float(dms.values[1])
    seconds = _ratio_to_float(dms.values[2])
    val = deg + minutes / 60.0 + seconds / 3600.0
    if str(ref.values).strip() in ("S", "W"):
        val = -val
    return val


_XMP_BLOCK_RE = re.compile(rb"<x:xmpmeta.*?</x:xmpmeta>", re.DOTALL)
_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _extract_xmp(jpg_path: str) -> bytes:
    with open(jpg_path, "rb") as f:
        data = f.read()
    m = _XMP_BLOCK_RE.search(data)
    return m.group(0) if m else b""


def _xmp_get_float(xmp: bytes, key: str) -> Optional[float]:
    pat = re.compile(rb"%s\s*=\s*\"([^\"]+)\"" % key.encode())
    m = pat.search(xmp)
    if m:
        s = m.group(1).decode(errors="ignore")
        fm = _FLOAT_RE.search(s)
        return float(fm.group()) if fm else None
    return None


# =========================
# METADATA INGEST
# =========================

def read_frame_meta_from_jpg(jpg_path: str) -> FrameMeta:
    with open(jpg_path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    if "GPS GPSLatitude" not in tags or "GPS GPSLongitude" not in tags:
        raise RuntimeError(f"No GPS data in {jpg_path}")

    lat = _dms_to_deg(tags["GPS GPSLatitude"], tags["GPS GPSLatitudeRef"])
    lon = _dms_to_deg(tags["GPS GPSLongitude"], tags["GPS GPSLongitudeRef"])

    alt_msl = None
    if "GPS GPSAltitude" in tags:
        a = tags["GPS GPSAltitude"].values[0]
        alt_msl = _ratio_to_float(a)

    xmp = _extract_xmp(jpg_path)

    yaw = _xmp_get_float(xmp, "Camera:Yaw")
    alt_agl = _xmp_get_float(xmp, "Camera:AboveGroundAltitude")

    if yaw is None:
        yaw = 0.0

    if alt_agl is None:
        if alt_msl is None:
            raise RuntimeError("No altitude found (XMP or EXIF)")
        alt_agl = alt_msl

    return FrameMeta(
        lat=float(lat),
        lon=float(lon),
        yaw_deg=float(yaw),
        alt_agl_m=float(alt_agl),
        alt_msl_m=(float(alt_msl) if alt_msl is not None else None),
    )


# =========================
# GEOREFERENCING CORE
# =========================

def _utm_crs(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_dict({"proj": "utm", "zone": zone, "south": lat < 0})


def thermal_pixel_to_lonlat(
    x_px: float,
    y_px: float,
    meta: FrameMeta,
) -> Tuple[float, float]:

    # Ground footprint
    hfov = math.radians(THERMAL_HFOV_DEG)
    vfov = math.radians(THERMAL_VFOV_DEG)

    alt_m = meta.alt_msl_m if meta.alt_msl_m is not None else meta.alt_agl_m

    ground_w_m = 2 * alt_m * math.tan(hfov / 2)
    ground_h_m = 2 * alt_m * math.tan(vfov / 2)

    mx = ground_w_m / THERMAL_WIDTH_PX
    my = ground_h_m / THERMAL_HEIGHT_PX

    cx = (THERMAL_WIDTH_PX - 1) / 2
    cy = (THERMAL_HEIGHT_PX - 1) / 2

    dx_m = (x_px - cx) * mx
    dy_m = (y_px - cy) * my

    yaw = math.radians(meta.yaw_deg)

    east_m  =  dx_m * math.cos(yaw) + dy_m * math.sin(yaw)
    north_m = -dx_m * math.sin(yaw) + dy_m * math.cos(yaw)

    utm = _utm_crs(meta.lon, meta.lat)
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    to_wgs = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)

    cx_m, cy_m = to_utm.transform(meta.lon, meta.lat)
    lon, lat = to_wgs.transform(cx_m + east_m, cy_m + north_m)

    return lon, lat


# =========================
# QUICK TEST
# =========================

if __name__ == "__main__":
    jpg = "IRX_1020.jpg"
    meta = read_frame_meta_from_jpg(jpg)
    print(meta)

    lon, lat = thermal_pixel_to_lonlat(320, 256, meta)
    print("center pixel:", lon, lat)

