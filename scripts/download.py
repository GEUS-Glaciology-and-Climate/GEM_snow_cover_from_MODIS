"""
Download MODIS CGF Snow Cover (MOD10A1F) for a configured site
==============================================================
Searches NASA Earthdata for granules covering the site AOI and
downloads them to the configured HDF directory.

Usage:
    python scripts/download.py --site zackenberg
"""

import argparse
import yaml
import geopandas as gpd
import earthaccess
from shapely.geometry import Polygon, MultiPolygon, LinearRing
from pathlib import Path

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True, help="Site name matching a config/<site>.yml")
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

# --- Load AOI in WGS84 ---
gdf = gpd.read_file(cfg["aoi"]).to_crs(epsg=4326)
geometry = gdf.unary_union

# --- Ensure counter-clockwise coordinates for earthaccess polygon search ---
def extract_coordinates_ccw(geom):
    if isinstance(geom, Polygon):
        coords = list(geom.exterior.coords)
    elif isinstance(geom, MultiPolygon):
        largest = max(geom.geoms, key=lambda g: g.area)
        coords = list(largest.exterior.coords)
    else:
        raise ValueError(f"Unsupported geometry type: {type(geom)}")
    if not LinearRing(coords).is_ccw:
        coords = coords[::-1]
    return coords

polygon_coords = extract_coordinates_ccw(geometry)

# --- Search and download ---
earthaccess.login(strategy="netrc")

print(f"Searching {cfg['modis_product']} granules for {cfg['site']} "
      f"({cfg['temporal_start']} to {cfg['temporal_end']})...")

results = earthaccess.search_data(
    short_name=cfg["modis_product"],
    temporal=(cfg["temporal_start"], cfg["temporal_end"]),
    polygon=polygon_coords,
)
print(f"Found {len(results)} granules")

hdf_dir = Path(cfg["hdf_dir"])
hdf_dir.mkdir(parents=True, exist_ok=True)

files = earthaccess.download(results, hdf_dir, threads=8)
print(f"Downloaded {len(files)} files to {hdf_dir}")
