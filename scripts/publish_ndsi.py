"""
Publish NDSI masked NetCDFs
===========================
Reads masked SCF NetCDFs, extracts the two masked variables, renames
snow_cover_fraction_masked -> ndsi_masked, strips verbose MODIS metadata,
and writes clean float32 files to publish/{site}/.

Output naming: publish/{site}/ndsi_{site}_{year}.nc

Usage:
    python scripts/publish_ndsi.py
"""
import xarray as xr
import numpy as np
from pathlib import Path

workingdir = Path("/home/shl/mdrev/projects/modis/snow_cover")

SITES = ["zackenberg", "nuuk", "disko"]

# Variable attrs to keep (drop all the verbose MODIS processing metadata)
KEEP_ATTRS = {"long_name", "valid_range", "grid_mapping", "mask_info", "Key"}

ENCODING_DEFAULTS = {"dtype": "float32", "_FillValue": np.nan}


def clean_attrs(attrs):
    return {k: v for k, v in attrs.items() if k in KEEP_ATTRS}


for site in SITES:
    src_dir = workingdir / f"netcdf/{site}_masked"
    out_dir = workingdir / f"publish/{site}"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob(f"{site}_scf_*.nc"))
    print(f"\n{site}: {len(files)} files")

    for fpath in files:
        year = fpath.stem.split("_")[-1]

        ds = xr.open_dataset(fpath)

        # Rename and select the two masked variables + spatial_ref
        ds_pub = ds[["snow_cover_fraction_masked", "cloud_persistence_masked", "spatial_ref"]].rename(
            {"snow_cover_fraction_masked": "ndsi_masked"}
        )

        # Strip verbose MODIS attrs from data variables
        for var in ["ndsi_masked", "cloud_persistence_masked"]:
            ds_pub[var].attrs = clean_attrs(ds_pub[var].attrs)

        # Update ndsi_masked long_name
        ds_pub["ndsi_masked"].attrs["long_name"] = "CGF NDSI Snow Cover (ice-free land only)"

        # Clean global attrs
        ds_pub.attrs = {
            "source":    ds.attrs.get("source", "MOD10A1F (NSIDC)"),
            "variables": "ndsi_masked, cloud_persistence_masked",
            "crs":       ds.attrs.get("crs", ""),
            "aoi":       ds.attrs.get("aoi", site.capitalize()),
            "history":   (
                "Derived from MOD10A1F CGF NDSI Snow Cover. "
                "Ocean and glacier pixels masked using land.shp and ice.shp. "
                "snow_cover_fraction_masked renamed to ndsi_masked. "
                "Unmasked variables and MODIS processing metadata removed."
            ),
        }

        out_path = out_dir / f"ndsi_{site}_{year}.nc"
        encoding = {
            "ndsi_masked":             ENCODING_DEFAULTS,
            "cloud_persistence_masked": ENCODING_DEFAULTS,
        }
        ds_pub.to_netcdf(out_path, encoding=encoding)
        ds.close()
        print(f"  wrote {out_path.name}")

print("\nDone.")
