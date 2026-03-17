"""
Clip and reproject MODIS SCF HDF granules to site AOI
======================================================
Reads daily HDF4 granules, mosaics tiles covering the same date,
reprojects to the site UTM CRS, clips to the AOI, and writes one
annual NetCDF per year.

Mosaicking is handled automatically: if multiple MODIS tiles cover
the AOI on a given date, they are merged before clipping.

Usage:
    python scripts/clip.py --site zackenberg
"""

import argparse
import re
import yaml
import geopandas as gpd
import rioxarray
import xarray as xr
from rioxarray.merge import merge_arrays
from pathlib import Path
from datetime import datetime
from itertools import groupby

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True, help="Site name matching a config/<site>.yml")
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site        = cfg["site"]
hdf_dir     = Path(cfg["hdf_dir"])
aoi_path    = cfg["aoi"]
target_epsg = cfg["target_epsg"]
out_dir     = Path(f"netcdf/{site}/")
out_dir.mkdir(parents=True, exist_ok=True)

variable_scf   = "CGF_NDSI_Snow_Cover"
variable_cloud = "Cloud_Persistence"

# --- Load AOI ---
aoi = gpd.read_file(aoi_path).to_crs(epsg=target_epsg)

# --- Helpers ---
def parse_date(path):
    m = re.search(r'\.A(\d{7})\.', path.name)
    if m:
        return datetime.strptime(m.group(1), "%Y%j")
    raise ValueError(f"Cannot parse date from {path.name}")

def open_native(hdf_path, variable):
    """Open a single HDF tile in its native MODIS Sinusoidal CRS."""
    return rioxarray.open_rasterio(
        f'HDF4_EOS:EOS_GRID:"{hdf_path}":MOD_Grid_Snow_500m:{variable}'
    )

def mosaic_and_clip(hdf_files, variable, aoi, target_epsg):
    """Open tiles in native CRS, mosaic, reproject once, then clip to AOI."""
    arrays = [open_native(f, variable) for f in hdf_files]
    merged = merge_arrays(arrays) if len(arrays) > 1 else arrays[0]
    reprojected = merged.rio.reproject(f"EPSG:{target_epsg}")
    clipped = reprojected.rio.clip(aoi.geometry, aoi.crs)
    return clipped.squeeze(drop=True)

# --- Group HDF files by year, then by date ---
hdf_files = sorted(hdf_dir.glob("*.hdf"), key=parse_date)

if not hdf_files:
    raise FileNotFoundError(f"No HDF files found in {hdf_dir}")

print(f"Found {len(hdf_files)} HDF granules in {hdf_dir}")

for year, year_iter in groupby(hdf_files, key=lambda f: parse_date(f).year):
    year_files = list(year_iter)

    # Group by date within the year
    date_groups: dict[datetime, list[Path]] = {}
    for f in year_files:
        date_groups.setdefault(parse_date(f), []).append(f)

    n_tiles_max = max(len(v) for v in date_groups.values())
    print(f"\n{year}: {len(date_groups)} dates, up to {n_tiles_max} tile(s) per date")

    arrays_scf, arrays_cloud, times = [], [], []
    ref_grid = None   # reference grid for reproject_match within the year

    for date, tiles in sorted(date_groups.items()):
        try:
            arr_scf   = mosaic_and_clip(tiles, variable_scf,   aoi, target_epsg)
            arr_cloud = mosaic_and_clip(tiles, variable_cloud, aoi, target_epsg)

            # Snap all days to the first day's grid to guarantee consistent shape
            if ref_grid is None:
                ref_grid = arr_scf
            else:
                arr_scf   = arr_scf.rio.reproject_match(ref_grid)
                arr_cloud = arr_cloud.rio.reproject_match(ref_grid)

            arrays_scf.append(arr_scf)
            arrays_cloud.append(arr_cloud)
            times.append(date)
        except Exception as e:
            print(f"  Skipping {date.date()} ({len(tiles)} tile(s)): {e}")
            continue

    if not arrays_scf:
        print(f"  No valid data for {year}, skipping")
        continue

    da_scf   = xr.concat(arrays_scf,   dim="time").assign_coords(time=times).sortby("time")
    da_cloud = xr.concat(arrays_cloud, dim="time").assign_coords(time=times).sortby("time")

    ds_out = da_scf.to_dataset(name="snow_cover_fraction")
    ds_out["cloud_persistence"] = da_cloud
    ds_out.attrs["source"]   = f"{cfg['modis_product']} (NSIDC)"
    ds_out.attrs["site"]     = site
    ds_out.attrs["crs"]      = f"EPSG:{target_epsg}"

    out_path = out_dir / f"{site}_scf_{year}.nc"
    ds_out.to_netcdf(out_path)
    print(f"  Wrote {out_path}")
