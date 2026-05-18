# simplet.py — Thermal Drone Ocean Analysis Pipeline

Disclaimer: Claude wrote this README. I have checked it for accuracy.

A processing pipeline for Autel 640T thermal drone imagery, designed to detect potential submarine groundwater discharge (SGD) signals in coastal ocean flights. The script processes a full flight directory, corrects within-frame thermal bias, identifies colder-than-baseline ocean regions, and aggregates those detections across frames into georeferenced raster and polygon outputs ready for area analysis in ArcGIS or QGIS.

---

## Quick Start

1. Set `OUTPUT_ROOT` near the top of the script to an empty folder where outputs should be written.
2. Set `FLIGHT_ROOT` at the bottom of the script to the directory containing the flight imagery.
3. Run:

```bash
python simplet.py
```

Everything else runs automatically. Inspect `byframe/passes/` for per-frame overlays, `layers/polygons/` for the final GeoJSON outputs, and `run_report.json` for a full log of every constant and per-frame result.

---

## Inputs

- **Flight directory** — a folder (or nested folder structure) containing paired thermal TIFFs (`IRX_####.TIFF`) and matching JPEG files (`IRX_####.jpg`). The JPEGs carry GPS, yaw, and altitude metadata via EXIF/XMP; the TIFFs carry the thermal pixel data.
- **`OUTPUT_ROOT`** — path where all outputs will be written (set near the top of the script).
- **`FLIGHT_ROOT`** — path to the flight directory (set at the bottom of the script in `main()`).

---

## What the Script Does

The pipeline runs in two passes over the flight imagery.

**Pass 1 — Bias field construction (all frames, no output written)**

For every frame in the flight that passes QC, the script computes a smooth-water ocean mask (S2) and accumulates a per-pixel running average of the frame's temperature deviation from its own S2 baseline. These deviations are averaged in image coordinates across all usable frames, filling sparse pixels with a local median, smoothing the result with a Gaussian filter, and anchoring it so the field is mean-zero. If `USE_DIRECTIONAL_BIAS` is enabled, this is done separately for each dominant yaw direction detected from the flight's yaw histogram, giving each heading its own bias field.

**Pass 2 — Frame processing and aggregation (selected frames)**

For each selected frame the script:

1. Reads and converts the thermal TIFF from raw DN to degrees Celsius.
2. Computes a local standard deviation texture map and builds a texture histogram.
3. Finds the angle-knee threshold that separates smooth ocean pixels (S) from textured land/surface-roughness pixels.
4. Applies QC filters on histogram metrics; frames that fail are saved to a cuts folder and skipped.
5. Reduces S to S2 — the primary smooth-ocean analysis mask — by keeping the largest connected component (or a variant; see `S2_MODE`).
6. Subtracts the bias field (if enabled) from the frame temperatures to correct the within-frame spatial bias.
7. Computes the ocean baseline as the `BASELINE_PERCENTILE`th percentile of corrected temperatures inside S2.
8. Computes `diff_c = corrected_temp − baseline` and builds cumulative cold masks at fixed temperature steps below the baseline (e.g. ≤ −0.20 °C, ≤ −0.40 °C, …, ≤ −0.60 °C), all restricted to S2.
9. Projects each frame's S2 and cold masks onto a shared UTM raster grid via metadata-derived georeferencing (center GPS, yaw, thermal FOV, ASL altitude). Each frame contributes at most one vote per grid cell.
10. Saves per-frame by-frame PNG overlays (S2, baseline band, temperature contours).

After all selected frames are processed, the script:

- Applies support-count and support-fraction filters to the aggregated rasters to suppress one-frame artifacts.
- Saves count, fraction, and final binary mask GeoTIFFs for each cold threshold and for S2 coverage.
- Polygonizes the final support-filtered masks and saves them as GeoJSON files.
- Saves the total flight footprint as a dissolved GeoJSON polygon.
- Saves bias field visualizations and a direction summary JSON.
- Writes a full JSON and text run report.

---

## Outputs

```
OUTPUT_ROOT/
├── byframe/
│   ├── passes/            # per-frame PNGs for frames that passed QC
│   │   ├── IRX_####_S_on_gray.png
│   │   ├── IRX_####_S2_on_gray.png
│   │   └── IRX_####_s2_temp_contours_on_gray.png
│   └── cuts/              # debug images for QC-rejected frames
│
├── bias_field/
│   ├── bias_global_smooth.png / bias_global_count.png
│   ├── bias_yaw_peak_###_smooth.png / bias_yaw_peak_###_count.png
│   ├── yaw_histogram_peaks.png
│   └── bias_direction_summary.json
│
├── layers/
│   ├── flight_footprint.geojson   # dissolved union of all frame footprints
│   ├── rasters/
│   │   ├── s2_count.tif           # frames with valid S2 coverage per cell
│   │   ├── cold_0p10_count.tif
│   │   ├── cold_0p10_fraction.tif # cold_count / s2_count
│   │   ├── cold_0p10_final_mask.tif
│   │   └── … (one set per threshold step)
│   └── polygons/
│       ├── s2_coverage_valid.geojson
│       ├── cold_0p10_final.geojson
│       └── … (one file per threshold step)
│
├── run_report.json         # full machine-readable run log with all constants and per-frame details
└── run_report.txt          # human-readable summary
```

The polygon GeoJSON files include an `area_m2` property computed in projected coordinates. The raster GeoTIFFs are in UTM with deflate compression and are ready to add directly as layers in ArcGIS or QGIS for area calculations and buffer intersections.

---

## Customization

### Paths (top of script)
| Constant | Description |
|---|---|
| `OUTPUT_ROOT` | Where all outputs are written |
| `FLIGHT_ROOT` | Flight directory passed to `main()` at the bottom |

### Frame selection
| Constant | Description |
|---|---|
| `BURST_SIZE` | Frames per burst group |
| `NUM_BURSTS` | Number of burst groups to process; set `BURST_SIZE` large enough to select the whole flight |

### Georeferencing
| Constant | Description |
|---|---|
| `THERMAL_HFOV_DEG` / `THERMAL_VFOV_DEG` | Thermal camera horizontal and vertical FOV — must match your camera model |
| `THERMAL_WIDTH_PX` / `THERMAL_HEIGHT_PX` | Thermal image resolution |
| `GEOREF_POSITION_SOURCE` | `"jpg"` (default) or `"tiff"` — which file's GPS coordinates to use for frame center position; yaw and altitude always come from the JPG/XMP |
| `HIGH_ALTITUDE_WARNING_M` | Prints a warning if any frame altitude exceeds this value (increase `AGG_GRID_RES_M` accordingly) |

### Bias field correction
| Constant | Description |
|---|---|
| `APPLY_BIAS_CORRECTION` | Toggle bias correction on/off |
| `BIAS_MIN_COUNT` | Minimum S2 observations required at a pixel before its bias estimate is used |
| `BIAS_SMOOTH_SIGMA` | Gaussian smoothing applied to the bias field, in pixels |
| `USE_DIRECTIONAL_BIAS` | Build separate bias fields per detected yaw direction; set `False` if bias does not appear yaw-dependent |
| `YAW_HIST_BIN_DEG` | Yaw histogram bin width for peak detection |
| `YAW_HIST_SMOOTH_SIGMA_BINS` | Smoothing applied to the yaw histogram before peak finding |
| `YAW_PEAK_MIN_DISTANCE_DEG` | Minimum angular separation between detected yaw peaks |
| `YAW_PEAK_SUPPORT_WINDOW_DEG` | Window around each peak used to count raw frame support |
| `YAW_PEAK_MIN_RAW_COUNT` | Minimum frames near a peak to form a directional bias group; should exceed `BIAS_MIN_COUNT` |

### Ocean mask (S / S2)
| Constant | Description |
|---|---|
| `TEX_WIN` | Local standard deviation window size (pixels) |
| `HIST_SMOOTH_K` | Kernel for main histogram smoothing; must be odd |
| `PEAK_HEIGHT_K` | Kernel for lighter smoothing used in the raw peak-height QC metric; must be odd |
| `ANGLE_DEG` / `ANGLE_RUN` | Controls the angle-knee threshold detection sensitivity |
| `S2_MODE` | `"largest"` keeps the biggest connected S component; `"edge"` keeps edge-touching components; `"within_x"` keeps pixels within `DILATE_PIXELS` of the largest component |
| `CONNECTIVITY_8` | 8- vs 4-neighbor connected components |
| `DILATE_PIXELS` | Distance threshold used only when `S2_MODE = "within_x"` |

### QC filters
Filters are toggled individually via the `FILTERS_ENABLED` dictionary. The threshold constants are:

| Constant | Description |
|---|---|
| `PEAK_HEIGHT_MIN` | Primary filter — minimum rawer texture peak height |
| `TEXTURE_THR_MAX` | Maximum texture threshold |
| `PEAK_X_MAX` | Maximum texture peak position |
| `PEAK_WIDTH_MAX` | Maximum peak FWHM width |
| `PEAK_SHARPNESS_MIN` | Minimum peak sharpness |
| `MIN_SMOOTH_FRAC` | Minimum fraction of the frame that must be S2 to compute a baseline |
| `S2_OVER_S_MIN` | Minimum S2/S ratio; rejects frames where the smooth-ocean mask fragments excessively |

### Baseline and cold masks
| Constant | Description |
|---|---|
| `BASELINE_PERCENTILE` | Percentile of corrected S2 temperatures used as the frame baseline |
| `BASELINE_BAND_DELTA_C` | Half-width of the baseline band drawn in red on the contour overlay |
| `CONTOUR_STEP_C` | Step size for the visual temperature contour PNG |
| `COLD_MASK_MIN_DELTA_C` | Step size for the cumulative cold masks used in aggregation (poorly named — this is a step size, not a minimum) |
| `COLD_MASK_MAX_DELTA_C` | The final cumulative threshold; the last mask includes all pixels colder than this value |

### Aggregation and support filtering
| Constant | Description |
|---|---|
| `AGG_GRID_RES_M` | Shared UTM raster cell size in meters; should be at or above the expected max thermal GSD (~0.4 m for these flights at 300–400 m altitude) |
| `MIN_S2_SUPPORT_COUNT` | Minimum number of frames that must show valid S2 coverage at a cell |
| `MIN_COLD_SUPPORT_COUNT` | Minimum number of frames that must mark a cell cold at a given threshold |
| `MIN_SUPPORT_FRACTION` | Minimum `cold_count / s2_count` ratio for a cell to survive in the final cold mask |

### Polygon output
| Constant | Description |
|---|---|
| `BUILD_AGGREGATE_POLYGONS` | Toggle polygon output on/off |
| `MIN_POLYGON_AREA_M2` | Remove polygons smaller than this area; `0.0` keeps all |

### Debug output
| Constant | Description |
|---|---|
| `VERBOSE_FRAME_LOGS` | Print per-frame details to terminal; set `False` for a cleaner progress bar |
| `SAVE_CUT_DEBUG` | Save debug images for QC-rejected frames |
| `SAVE_DEBUG_PROJECTED_S2` | Save individual projected S2 GeoTIFFs for specific frames (slow) |
| `DEBUG_PROJECTED_S2_FRAME_IDS` | Set of frame ID strings to save projected S2 debug rasters for |
| `SAVE_DIST_HEATMAP` | Save distance-to-largest-component heatmaps (only meaningful when `S2_MODE = "within_x"`) |

---

## Dependencies

```
numpy>=1.26
pyproj>=3.6
exifread>=3.0
rasterio>=1.3.10
scipy>=1.11
simplekml>=1.3.6
Pillow>=10.0
shapely>=2.0
matplotlib
rich>=13.0
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Known Limitations

- **Whole-flight georeferencing offset.** Entire flights are often shifted by a steady offset of a few meters relative to true ground position, meaning the cumulative S2 coverage area and cold-mask polygons may appear slightly displaced on a basemap. This is a known limitation of metadata-based georeferencing without ground control points.
- **Georeferencing is approximate within a flight.** Pixel projection uses center GPS, metadata yaw, thermal FOV, and ASL altitude under a flat-nadir model. Frame-to-frame offsets are not consistent, so outputs should be treated as approximate spatial products rather than survey-grade boundaries.
- **Pitch and roll are not modeled.** The projection assumes the camera looks straight down. Small pitch or roll errors shift edge pixels by a meter or more.
- **Flat water-plane assumption.** All pixels in a frame are projected onto a single plane at the altitude-implied height. Frame footprints near cliffs or steep terrain will look distorted.
- **Shaded rock inclusion.** Because ocean segmentation is texture-based, thermally smooth shaded rocks can be misclassified as ocean. Small shaded rock fragments connected to the water body may survive into S2 and appear in aggregated outputs. The support-count and support-fraction filtering (`MIN_S2_SUPPORT_COUNT`, `MIN_COLD_SUPPORT_COUNT`, `MIN_SUPPORT_FRACTION`) reduces but does not fully eliminate this.
- **Large shaded regions and segmentation failure.** When large cliffs are present and the sun angle produces extensive shadow, the shaded area can be smooth enough in texture to dominate the S2 mask and displace the ocean component. The `S2_OVER_S_MIN` filter rejects the worst cases, but heavily shaded frames should be identified and cut from the flight before processing.
- **Baseline is frame-relative.** Each frame's baseline is the S2 percentile of its own corrected temperatures. Cold anomaly detections are relative to each frame's ocean state, not to an absolute calibrated temperature.
- **Bias correction depends on S2 quality.** If the ocean mask is noisy or covers too little of the image, the bias field will be unreliable. Directional grouping helps but requires enough frames per yaw direction.
