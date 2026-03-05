import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LinearRing
import earthaccess
from pathlib import Path
from datetime import date

earthaccess.login(strategy="netrc")

# Load and reproject
gdf = gpd.read_file("shp/zackenberg_aoi.gpkg").to_crs(epsg=4326)

# Combine into one geometry
geometry = gdf.unary_union

# Ensure counter-clockwise coordinates
def extract_coordinates_ccw(geom):
    if isinstance(geom, Polygon):
        coords = list(geom.exterior.coords)
    elif isinstance(geom, MultiPolygon):
        largest = max(geom.geoms, key=lambda g: g.area)
        coords = list(largest.exterior.coords)
    else:
        raise ValueError("Unsupported geometry type")

    if not LinearRing(coords).is_ccw:
        coords = coords[::-1]

    return coords

polygon_coords = extract_coordinates_ccw(geometry)

# Search

results = earthaccess.search_data(
    short_name='MOD10A1F',
    temporal=('2000-02-24', date.today().isoformat()),
    polygon=polygon_coords
)
out_path = Path('/mnt/glaciologi/GEM/remote_sensing_and_modelling/data/MODIS_CGF_SCF/zackenberg')
out_path.mkdir(parents=True, exist_ok=True)

# Earthaccess api downloads only the files that are not already downloaded
files = earthaccess.download(results, str(out_path), threads=8)

