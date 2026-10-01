"""Read the per-year GeoTIFFs exported by scripts/gee_export.js.

Each file holds 46 composites x 8 bands named YYYY_MM_DD_<channel>, channels
T_day, T_night, A_day, A_night (LST, deg C) and *_err (MODIS error class).
"""
import glob, os, re
import numpy as np

CHANNELS = ["T_day", "T_night", "A_day", "A_night",
            "T_dayerr", "T_nighterr", "A_dayerr", "A_nighterr"]


def load_tile(folder, prefix):
    """Return dict channel -> (n_pixels, n_composites) float32 matrix over land
    pixels, plus the list of composite dates. Composites missing for a channel
    (e.g. Aqua 2022-04-07) are filled with NaN so all channels share one date axis."""
    import rasterio
    files = sorted(glob.glob(os.path.join(folder, f"{prefix}_*.tif")))
    if not files:
        raise FileNotFoundError(f"no {prefix}_*.tif in {folder}")
    raw = {}
    for f in files:
        with rasterio.open(f) as r:
            arr = r.read()
            for i, name in enumerate(r.descriptions):
                m = re.match(r"(\d{4}_\d\d_\d\d)_(.*)", name)
                raw[(m.group(1), m.group(2))] = arr[i]
    dates = sorted({k[0] for k in raw})
    shape = next(iter(raw.values())).shape
    cube = {}
    for ch in CHANNELS:
        cube[ch] = np.stack([raw.get((d, ch), np.full(shape, np.nan, np.float32))
                             for d in dates], -1).astype(np.float32)
    land = np.isfinite(cube["T_day"]).any(-1) | np.isfinite(cube["A_day"]).any(-1)
    return {ch: v[land] for ch, v in cube.items()}, dates
