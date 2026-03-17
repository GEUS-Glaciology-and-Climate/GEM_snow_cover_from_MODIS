"""
ArcticDEM Terrain Extraction
=============================
Loads the ArcticDEM Mosaic v4.1 (32 m) from AWS S3 as a Cloud Optimized
GeoTIFF, clips to the site AOI, and computes elevation, slope, and aspect.
Output is a single NetCDF for use in downstream spatial analysis.

The DEM is streamed directly from the public S3 bucket — no local copy needed.

Usage:
    python scripts/get_terrain.py --site zackenberg
"""

import argparse
import os
import yaml
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray
import pyproj
from pathlib import Path

# Anonymous S3 access via GDAL virtual filesystem
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
# Allow pyproj to fetch EGM2008 geoid grid on demand
pyproj.network.set_network_enabled(True)

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True, help="Site name matching a config/<site>.yml")
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site        = cfg["site"]
aoi_path    = cfg["aoi"]
target_epsg = cfg["target_epsg"]
land_shp    = cfg["land_shp"]
out_dir     = Path(f"netcdf/{site}/")
out_dir.mkdir(parents=True, exist_ok=True)

# ArcticDEM Mosaic v4.1, 32 m — public AWS S3 (no auth required)
# The VRT references all individual tiles and acts as a seamless mosaic
DEM_URL = "/vsis3/pgc-opendata-dems/arcticdem/mosaics/v4.1/32m_dem_tiles.vrt"

# --- Load AOI and land mask ---
aoi  = gpd.read_file(aoi_path)
land = gpd.read_file(land_shp).to_crs(epsg=target_epsg)

# --- Stream DEM from S3 ---
print("Opening ArcticDEM COG from S3 (this may take a moment)...")
dem_raw = rioxarray.open_rasterio(
    DEM_URL,
    masked=True,
    chunks="auto",
)
print(f"  DEM CRS      : {dem_raw.rio.crs}")
print(f"  DEM resolution: {dem_raw.rio.resolution()} m")

# Reproject AOI to DEM native CRS for windowed read
aoi_dem_crs = aoi.to_crs(dem_raw.rio.crs)

# Clip to bounding box first (lazy — only fetches needed COG tiles)
print("Clipping to AOI bounding box...")
dem_clipped = dem_raw.rio.clip_box(*aoi_dem_crs.total_bounds)
dem_clipped = dem_clipped.compute()   # trigger the actual S3 fetch

# Reproject to site UTM CRS
print(f"Reprojecting to EPSG:{target_epsg}...")
dem_proj = dem_clipped.rio.reproject(f"EPSG:{target_epsg}")

# Clip precisely to AOI polygon
aoi_proj = aoi.to_crs(epsg=target_epsg)
dem_proj = dem_proj.rio.clip(aoi_proj.geometry, aoi_proj.crs)

# Mask ocean: keep only land pixels (glaciated areas included)
print("Masking ocean...")
dem_proj = dem_proj.rio.clip(land.geometry, land.crs, drop=False)
dem_proj = dem_proj.squeeze(drop=True)

elev = dem_proj.values.astype(np.float32)
print(f"  Grid size    : {elev.shape}")
print(f"  Elevation    : {np.nanmin(elev):.1f} – {np.nanmax(elev):.1f} m (ellipsoidal)")

# Save 32 m DEM at this stage (land + ocean, full bounding box) so that the
# shadow-geometry script has surrounding terrain for horizon angle computation.
dem_32m_path = out_dir / f"{site}_dem_32m.tif"

# --- Geoid correction: ellipsoidal → orthometric (EGM2008) ---
print("Applying EGM2008 geoid correction...")
xx, yy = np.meshgrid(dem_proj.x.values, dem_proj.y.values)

# UTM → WGS84 geographic
utm_to_wgs84 = pyproj.Transformer.from_crs(
    f"EPSG:{target_epsg}", "EPSG:4326", always_xy=True
)
lon_flat, lat_flat = utm_to_wgs84.transform(xx.ravel(), yy.ravel())

# WGS84 ellipsoidal (EPSG:4979) → EGM2008 orthometric (EPSG:3855)
# Pass z=0 to recover geoid undulation N at each point: z_out = -N
ellips_to_egm = pyproj.Transformer.from_crs(
    "EPSG:4979", "EPSG:3855", always_xy=True
)
_, _, z_out = ellips_to_egm.transform(lon_flat, lat_flat, np.zeros(len(lon_flat)))
N = -z_out.reshape(elev.shape).astype(np.float32)

elev = np.where(np.isfinite(elev), elev - N, np.nan)
print(f"  Geoid undulation: {np.nanmean(N):.2f} m (mean over AOI)")
print(f"  Elevation    : {np.nanmin(elev):.1f} – {np.nanmax(elev):.1f} m (orthometric, EGM2008)")

# Write 32 m DEM (geoid-corrected, before ocean masking) as GeoTIFF
dem_proj_corrected = dem_proj.copy(data=elev)
dem_proj_corrected.rio.to_raster(dem_32m_path, dtype="float32")
print(f"  Saved 32 m DEM: {dem_32m_path}")

# --- Slope and aspect ---
print("Computing slope and aspect...")
res_x, res_y = dem_proj.rio.resolution()   # res_y is negative for north-up

# np.gradient returns [d/d_row, d/d_col]
# Rows increase southward, so d/d_row points south → negate for north convention
dz_drow, dz_dcol = np.gradient(elev, abs(res_y), abs(res_x))

# Slope: steepest descent magnitude, in degrees
slope = np.degrees(np.arctan(np.sqrt(dz_dcol**2 + dz_drow**2)))

# Aspect: compass bearing of downslope direction (0 = N, 90 = E, clockwise)
# arctan2(east_component, north_component)
# east component  =  dz/dx  (dz_dcol, positive eastward)
# north component = -dz/dy_north = +dz/drow (rows increase southward)
aspect = np.degrees(np.arctan2(dz_dcol, -dz_drow))
aspect = aspect % 360

# --- Build output dataset ---
# Use coordinates from the reprojected DataArray so spatial metadata carries through
coords = {
    "y": ("y", dem_proj.y.values, {
        "units": "m",
        "standard_name": "projection_y_coordinate",
        "long_name": "y coordinate of projection",
    }),
    "x": ("x", dem_proj.x.values, {
        "units": "m",
        "standard_name": "projection_x_coordinate",
        "long_name": "x coordinate of projection",
    }),
}

ds_out = xr.Dataset(
    {
        "elevation": (["y", "x"], elev),
        "slope":     (["y", "x"], slope.astype(np.float32)),
        "aspect":    (["y", "x"], aspect.astype(np.float32)),
    },
    coords=coords,
)

ds_out["elevation"].attrs = {"units": "m",       "long_name": "Elevation above EGM2008 geoid (orthometric)", "grid_mapping": "spatial_ref"}
ds_out["slope"].attrs     = {"units": "degrees", "long_name": "Terrain slope",             "grid_mapping": "spatial_ref"}
ds_out["aspect"].attrs    = {"units": "degrees", "long_name": "Terrain aspect (0=N, clockwise)", "grid_mapping": "spatial_ref"}

# Write CRS as a proper grid_mapping variable (CF convention)
ds_out = ds_out.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs(
    f"EPSG:{target_epsg}", grid_mapping_name="spatial_ref"
)

ds_out.attrs["source"]    = "ArcticDEM Mosaic v4.1 (32 m), Polar Geospatial Center; geoid correction: EGM2008 via pyproj"
ds_out.attrs["site"]      = site
ds_out.attrs["dem_url"]   = DEM_URL.replace("/vsis3/", "s3://")

out_path = out_dir / f"{site}_terrain.nc"
ds_out.to_netcdf(out_path)

print(f"\nSaved: {out_path}")
print(f"  Slope  : {np.nanmin(slope):.1f} – {np.nanmax(slope):.1f} °")
print(f"  Aspect : 0 – 360 °")
