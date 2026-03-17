"""
Mask MODIS SCF NetCDFs with land and ice shapefiles
====================================================
Reads each annual NetCDF, rasterises the land and ice masks
onto the SCF grid, and writes two sets of masked NetCDFs:

  {site}_masked/          — ice-free land only (ocean + glaciers masked)
  {site}_masked_withice/  — all land including glaciers (ocean only masked)

Both retain the variable name snow_cover_fraction_masked so downstream
scripts work unchanged against either directory.

Usage:
    python scripts/mask_scf.py --site zackenberg
"""

import argparse
import yaml
import xarray as xr
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import Affine
from pathlib import Path

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True, help="Site name matching a config/<site>.yml")
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site              = cfg["site"]
nc_dir            = Path(f"netcdf/{site}/")
out_nc_dir        = Path(f"netcdf/{site}_masked/")
out_nc_dir_withice = Path(f"netcdf/{site}_masked_withice/")
out_nc_dir.mkdir(parents=True, exist_ok=True)
out_nc_dir_withice.mkdir(parents=True, exist_ok=True)

land_shp = Path(cfg["land_shp"])
ice_shp  = Path(cfg["ice_shp"])

target_epsg = cfg["target_epsg"]
variable    = "snow_cover_fraction"

# --- Load and reproject shapefiles ---
print("Loading and reprojecting shapefiles...")
land = gpd.read_file(land_shp).to_crs(epsg=target_epsg)
ice = gpd.read_file(ice_shp).to_crs(epsg=target_epsg)

# --- Helper: rasterise a GeoDataFrame onto a given grid ---
def make_mask(gdf, x_coords, y_coords):
    """Rasterise polygons onto a grid defined by x and y coordinate arrays.
    Returns a boolean array (True = inside polygons)."""
    # Build affine transform from the coordinate arrays
    res_x = float(x_coords[1] - x_coords[0])
    res_y = float(y_coords[1] - y_coords[0])  # typically negative
    transform = Affine(res_x, 0, float(x_coords[0]) - res_x / 2,
                       0, res_y, float(y_coords[0]) - res_y / 2)

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        return np.zeros((len(y_coords), len(x_coords)), dtype=bool)

    mask = rasterize(
        shapes,
        out_shape=(len(y_coords), len(x_coords)),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    return mask.astype(bool)

# --- Process each annual NetCDF ---
nc_files = sorted(nc_dir.glob(f"{site}_scf_*.nc"))
if not nc_files:
    raise FileNotFoundError(f"No NetCDF files found in {nc_dir}")

print(f"Found {len(nc_files)} files to process\n")

# Rasterise masks once using the grid from the first file
ref_ds = xr.open_dataset(nc_files[0])
ref_da = ref_ds[variable]
x_coords = ref_da.x.values
y_coords = ref_da.y.values
ref_ds.close()

print("Rasterising land mask...")
land_mask = make_mask(land, x_coords, y_coords)
print(f"  Land pixels: {land_mask.sum()} / {land_mask.size}")

print("Rasterising ice mask...")
ice_mask = make_mask(ice, x_coords, y_coords)
print(f"  Ice pixels: {ice_mask.sum()} / {land_mask.sum()} land pixels")

# Mask 1: ice-free land only (ocean + glaciers masked)
valid_mask = land_mask & ~ice_mask
# Mask 2: all land including glaciers (ocean only masked)
valid_mask_withice = land_mask
print(f"  Valid (ice-free land) pixels:    {valid_mask.sum()}")
print(f"  Valid (land + glacier) pixels:   {valid_mask_withice.sum()}\n")

# --- Apply masks to each file ---
def apply_mask(ds, mask, scf_long_name, mask_info):
    """Return a new dataset with snow_cover_fraction_masked and cloud_persistence_masked."""
    ds_out = ds.copy()
    ds_out["snow_cover_fraction_masked"] = ds[variable].where(mask)
    ds_out["snow_cover_fraction_masked"].attrs = ds[variable].attrs.copy()
    ds_out["snow_cover_fraction_masked"].attrs["long_name"] = scf_long_name
    ds_out["snow_cover_fraction_masked"].attrs["mask_info"] = mask_info

    ds_out["cloud_persistence_masked"] = ds["cloud_persistence"].where(mask)
    ds_out["cloud_persistence_masked"].attrs = ds["cloud_persistence"].attrs.copy()
    ds_out["cloud_persistence_masked"].attrs["long_name"] = (
        "Cloud Persistence " + scf_long_name.split("(")[1].rstrip(")")
    )
    ds_out["cloud_persistence_masked"].attrs["mask_info"] = mask_info
    return ds_out

for nc_path in nc_files:
    print(f"Processing {nc_path.name}...")
    ds = xr.open_dataset(nc_path)

    # Version 1: ice-free land only
    ds_masked = apply_mask(
        ds, valid_mask,
        scf_long_name="NDSI Snow Cover (ice-free land only)",
        mask_info="Ocean and glaciers masked using land.shp and ice.shp",
    )
    out_path = out_nc_dir / nc_path.name
    ds_masked.to_netcdf(out_path)
    print(f"  -> {out_path}")

    # Version 2: land including glaciers
    ds_withice = apply_mask(
        ds, valid_mask_withice,
        scf_long_name="NDSI Snow Cover (land including glaciers)",
        mask_info="Ocean masked using land.shp; glaciers retained",
    )
    out_path_wi = out_nc_dir_withice / nc_path.name
    ds_withice.to_netcdf(out_path_wi)
    print(f"  -> {out_path_wi}")

    ds.close()

print("\nDone. Original files untouched in:", nc_dir)
print("Ice-free land files written to:   ", out_nc_dir)
print("Land + glacier files written to:  ", out_nc_dir_withice)
