import xarray as xr
import rioxarray
from pathlib import Path
from datetime import datetime
import geopandas as gpd
import re

# --- Config ---
aoi_path = "shp/zackenberg_aoi.gpkg"
hdf_dir = Path("/mnt/glaciologi/GEM/remote_sensing_and_modelling/data/MODIS_CGF_SCF/zackenberg/")
out_dir = Path("./netcdf/zackenberg/")
target_epsg = 32627  # UTM 27N for Zackenberg
variable_scf = "CGF_NDSI_Snow_Cover"  # the gap-filled SCF band
variable_cloud = "Cloud_Persistence"   # cloud gap length band

out_dir.mkdir(parents=True, exist_ok=True)

# --- Load AOI in target CRS ---
aoi = gpd.read_file(aoi_path).to_crs(epsg=target_epsg)

# --- Parse date from filename ---
def date_from_filename(path):
    # MOD10A1F filenames contain e.g. .A2024001. (YYYYDDD)
    m = re.search(r'\.A(\d{7})\.', path.name)
    if m:
        return datetime.strptime(m.group(1), "%Y%j")
    raise ValueError(f"Cannot parse date from {path.name}")

# --- Process one HDF file for a given variable ---
def process_granule(hdf_path, variable, aoi_geom, target_epsg):
    ds = rioxarray.open_rasterio(
        f'HDF4_EOS:EOS_GRID:"{hdf_path}":MOD_Grid_Snow_500m:{variable}'
    )
    # Reproject to target UTM
    ds = ds.rio.reproject(f"EPSG:{target_epsg}")
    # Clip to AOI
    ds = ds.rio.clip(aoi.geometry, aoi.crs)
    return ds.squeeze(drop=True)

# --- Group files by year and build annual NetCDFs ---
hdf_files = sorted(hdf_dir.glob("*.hdf"), key=date_from_filename)

# Group by year
from itertools import groupby
for year, files in groupby(hdf_files, key=lambda f: date_from_filename(f).year):
    file_list = list(files)
    print(f"Processing {year}: {len(file_list)} granules")

    arrays_scf = []
    arrays_cloud = []
    times = []
    for f in file_list:
        try:
            arr_scf = process_granule(f, variable_scf, aoi.geometry, target_epsg)
            arr_cloud = process_granule(f, variable_cloud, aoi.geometry, target_epsg)
            arrays_scf.append(arr_scf)
            arrays_cloud.append(arr_cloud)
            times.append(date_from_filename(f))
        except Exception as e:
            print(f"  Skipping {f.name}: {e}")
            continue

    if not arrays_scf:
        continue

    # Stack along time dimension
    da_scf = xr.concat(arrays_scf, dim="time")
    da_cloud = xr.concat(arrays_cloud, dim="time")
    da_scf["time"] = times
    da_cloud["time"] = times
    da_scf = da_scf.sortby("time")
    da_cloud = da_cloud.sortby("time")

    # Convert to dataset and add metadata
    ds_out = da_scf.to_dataset(name="snow_cover_fraction")
    ds_out["cloud_persistence"] = da_cloud
    ds_out.attrs["source"] = "MOD10A1F (NSIDC)"
    ds_out.attrs["variables"] = f"{variable_scf}, {variable_cloud}"
    ds_out.attrs["crs"] = f"EPSG:{target_epsg}"
    ds_out.attrs["aoi"] = "Zackenberg"

    out_path = out_dir / f"zackenberg_scf_{year}.nc"
    ds_out.to_netcdf(out_path)
    print(f"  Wrote {out_path}")
