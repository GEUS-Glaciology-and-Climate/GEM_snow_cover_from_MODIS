# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Install dependencies with conda:

```bash
conda env create -f environment.yml
conda activate modis_scf
```

Key packages: `xarray`, `rioxarray`, `geopandas`, `earthaccess`, `pymannkendall`, `pingouin`, `statsmodels`.

## Running Scripts

All scripts are run directly from the project root:

```bash
python scripts/<script_name>.py
```

Scripts expect to be run from `/home/shl/mdrev/projects/modis/snow_cover` — several use hardcoded absolute paths (e.g. `workingdir = Path("/home/shl/mdrev/projects/modis/snow_cover")`).

## Pipeline Architecture

The analysis follows a linear pipeline focused on Zackenberg (one of three GEM sites: Zackenberg, Nuuk, Disko):

### 1. Download — `scripts/download_all.py`
Downloads MOD10A1F (MODIS CGF NDSI Snow Cover) HDF files from NASA Earthdata using `earthaccess`. Requires a `.netrc` file with Earthdata credentials. Raw HDF files go to `/mnt/glaciologi/GEM/remote_sensing_and_modelling/data/MODIS_CGF_SCF/`.

### 2. Clip & Convert — `scripts/clip_zackenberg.py`
Reads HDF files using `rioxarray` with the subdataset path `HDF4_EOS:EOS_GRID:"<file>":MOD_Grid_Snow_500m:<variable>`. Reprojects to UTM 27N (EPSG:32627), clips to AOI shapefile, and writes annual NetCDFs to `netcdf/zackenberg/zackenberg_scf_<year>.nc`. Extracts two variables: `CGF_NDSI_Snow_Cover` and `Cloud_Persistence`.

### 3. Mask — `scripts/mask_scf.py`
Rasterises land/ice shapefiles from `/home/shl/mdrev/data/sdfi/G250_VEKTOR/` onto the SCF grid, producing `snow_cover_fraction_masked` (ice-free land only) in `netcdf/zackenberg_masked/`.

### 4. Snow-Free Days — `scripts/get_snow_free_days.py`
Computes mean snow-free days per year across NDSI thresholds 10–70%. Flag values >100 are excluded. Years with <320 observations are skipped. Outputs: `results/csvs/snow_free_days.csv`, figures in `figures/zackenberg/`.

### 5. Climate Predictors — `scripts/derive_climate_predictors.py`
Derives from CARRA reanalysis NetCDFs (`data/CARRA/`):
- **PDD**: sum of positive daily mean temperatures (calendar year)
- **Winter precip**: cold-season (T < 0°C) precipitation, hydrological year ending September
- **Melt days**: days with daily mean T > −3°C

Also loads in situ temperature from GEM station data for PDD validation. Outputs: `results/csvs/climate_predictors_zackenberg.csv`.

CARRA data was extracted at nearest PROMICE station points using `data/CARRA/data-extraction-nearest-point.py` (reads GRIB files from `/mnt/ice/CARRA/`).

### 6. Analysis — `scripts/analyse_*.py`
- `analyse_trends.py`: Mann-Kendall test + Sen's slope on snow-free days time series
- `analyse_driver_trends.py`: Same tests on PDD, winter precip, and snow-free days together
- `analyse_correlations.py`: Pearson correlation matrix, partial correlations (pingouin), and VIF
- `analyse_regression.py`: OLS regression (`snow_free_days ~ PDD + winter_precip`) with residual diagnostics

All analysis scripts write figures to `figures/zackenberg/`, CSVs to `results/csvs/`, and LaTeX tables alongside the figures.

## Data Layout

```
data/CARRA/          # CARRA reanalysis NetCDFs (t2m, tp) at PROMICE station locations
shp/                 # AOI polygons and station positions (UTM and WGS84 variants)
netcdf/zackenberg/           # Annual SCF NetCDFs (raw clips)
netcdf/zackenberg_masked/    # Annual SCF NetCDFs (land/ice masked)
results/csvs/        # Intermediate and final CSVs
figures/zackenberg/  # Output plots and LaTeX tables
```

## Key Conventions

- NDSI threshold of **40%** is the primary threshold used in trend/regression analysis (sensitivity tested across 10–70%).
- MODIS flag values (200–255) represent missing/cloud/water — only values ≤100 are valid NDSI.
- Hydrological year runs **October 1 to September 30** for winter precipitation (`YE-SEP` in pandas/xarray = year ending September 30).
- Statistical significance threshold: **α = 0.05** throughout.
- Outputs include both PNG figures and LaTeX `.tex` table snippets for direct inclusion in papers.
