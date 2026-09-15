from pathlib import Path
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt


input_tiff = Path("/Volumes/EXTERNAL HD/July 2024/Autel-West of Vaihu/103MEDIA/IRX_0971.TIFF")
output_png = Path("output.png")


def to_celsius_autel(raw):
    return raw.astype(np.float32) * 0.1 - 273.15


def normalize_for_display(img):
    finite = np.isfinite(img)
    v = img[finite]

    lo = np.percentile(v, 2)
    hi = np.percentile(v, 98)

    if hi <= lo:
        hi = lo + 1.0

    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


raw = tiff.imread(input_tiff)
raw = np.squeeze(raw)

temp_c = to_celsius_autel(raw)
disp = normalize_for_display(temp_c)

plt.figure(figsize=(8, 6))
plt.imshow(disp, cmap="gray", vmin=0.0, vmax=1.0)
plt.axis("off")
plt.savefig(output_png, dpi=150, bbox_inches="tight", pad_inches=0)
plt.close()

print(f"saved {output_png}")
print(f"raw range: {raw.min()} to {raw.max()}")
print(f"celsius range: {np.nanmin(temp_c)} to {np.nanmax(temp_c)}")