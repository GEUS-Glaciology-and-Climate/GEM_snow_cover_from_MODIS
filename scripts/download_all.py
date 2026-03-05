import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LinearRing
import earthaccess

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
    temporal=('2020-02-24', '2025-12-31'),
    polygon=polygon_coords
)

# Download
files = earthaccess.download(results, '/mnt/glaciologi/GEM/remote_sensing_and_modelling/data/MODIS_CGF_SCF', threads=8))
