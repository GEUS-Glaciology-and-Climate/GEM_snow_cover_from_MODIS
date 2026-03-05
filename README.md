# MODIS Snow cover fraction at GEM sites

Extract MODIS cloud gap filled snow cover fraction for the three GEM sites to generate snow free days.

## Requirements

    conda env create -f environment.yml
    conda activate modis_scf

## Usage

Run scripts individually from the project root in pipeline order:

    python scripts/download_all.py
    python scripts/clip_zackenberg.py
    python scripts/mask_scf.py
    python scripts/get_snow_free_days.py
    python scripts/derive_climate_predictors.py
    python scripts/analyse_trends.py
    python scripts/analyse_driver_trends.py
    python scripts/analyse_correlations.py
    python scripts/analyse_regression.py

## Project Structure

    ├── environment.yml
    ├── scripts/
    │   ├── download_all.py              # Download MOD10A1F HDF files via earthaccess
    │   ├── clip_zackenberg.py           # Reproject, clip to AOI, write annual NetCDFs
    │   ├── mask_scf.py                  # Apply land/ice masks
    │   ├── get_snow_free_days.py        # Compute snow-free days across NDSI thresholds
    │   ├── derive_climate_predictors.py # Compute PDD, winter precip, melt days from CARRA
    │   ├── load_climate_drivers.py      # Helper for loading CARRA climate data
    │   ├── analyse_trends.py            # Mann-Kendall trend test on snow-free days
    │   ├── analyse_driver_trends.py     # Trends in PDD, winter precip, and snow-free days
    │   ├── analyse_correlations.py      # Pearson and partial correlations + VIF
    │   └── analyse_regression.py        # OLS regression with residual diagnostics
    ├── data/
    │   └── CARRA/                       # CARRA reanalysis NetCDFs (t2m, tp at stations)
    ├── shp/                             # AOI polygons and station positions
    ├── netcdf/
    │   ├── zackenberg/                  # Annual SCF NetCDFs (clipped)
    │   └── zackenberg_masked/           # Annual SCF NetCDFs (land/ice masked)
    ├── results/
    │   └── csvs/                        # Intermediate and final CSVs
    └── figures/
        └── zackenberg/                  # Output plots and LaTeX tables

## Notes

- Earthdata login credentials must be stored in `~/.netrc` for `download_all.py`.
- Raw HDF files are downloaded to `/mnt/glaciologi/GEM/remote_sensing_and_modelling/data/MODIS_CGF_SCF/`.
- Land/ice shapefiles for masking are read from `/home/shl/mdrev/data/sdfi/G250_VEKTOR/`.
- In situ temperature data (GEM Zackenberg station) is read from `/home/shl/mdrev/data/GEM/zackenberg/`.
- NDSI values >100 are flag values (cloud, water, missing) and are excluded from analysis.
- The primary NDSI snow-free threshold is 40%; sensitivity is tested across 10–70%.
- Years with fewer than 320 daily observations are excluded from snow-free day counts.
