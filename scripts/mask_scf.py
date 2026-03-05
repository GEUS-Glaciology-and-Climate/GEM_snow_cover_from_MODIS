"""
Mask MODIS SCF NetCDFs with land and ice shapefiles
====================================================
Reads each annual NetCDF, rasterises the land and ice masks
onto the SCF grid, applies them, and writes new NetCDFs with
both the original and masked variables.

Logic:
    - Keep pixels inside land polygons
    - Remove pixels inside ice polygons
    - Result: snow_cover_fraction_masked = SCF for ice-free land only

Usage:
    python mask_scf.py
"""

import xarray as xr
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import Affine
from pathlib import Path

# --- Config ---
nc_dir = Path("./netcdf/zackenberg/")
out_dir = Path("./netcdf/zackenberg_masked/")
out_dir.mkdir(parents=True, exist_ok=True)

land_shp = Path("/home/shl/mdrev/data/sdfi/G250_VEKTOR/Land.shp")
ice_shp = Path("/home/shl/mdrev/data/sdfi/G250_VEKTOR/Is.shp")

target_epsg = 32627  # Zackenberg UTM zone
mask_epsg = 32624    # Source CRS of the shapefiles
variable = "snow_cover_fraction"

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
nc_files = sorted(nc_dir.glob("zackenberg_scf_*.nc"))
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

# Combined mask: True = valid (land AND not ice)
valid_mask = land_mask & ~ice_mask
print(f"  Valid (ice-free land) pixels: {valid_mask.sum()}\n")

# --- Apply mask to each file ---
for nc_path in nc_files:
    print(f"Processing {nc_path.name}...")
    ds = xr.open_dataset(nc_path)

    # Apply mask: set pixels outside valid area to NaN
    masked = ds[variable].where(valid_mask)

    # Add as new variable
    ds["snow_cover_fraction_masked"] = masked
    ds["snow_cover_fraction_masked"].attrs = ds[variable].attrs.copy()
    ds["snow_cover_fraction_masked"].attrs["long_name"] = (
        "NDSI Snow Cover (ice-free land only)"
    )
    ds["snow_cover_fraction_masked"].attrs["mask_info"] = (
        "Masked to ice-free land using land.shp and is.shp"
    )

    out_path = out_dir / nc_path.name
    ds.to_netcdf(out_path)
    ds.close()
    print(f"  -> {out_path}")

print("\nDone. Original files untouched in:", nc_dir)
print("Masked files written to:", out_dir)
