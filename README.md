# MODIS Snow Cover Fraction at GEM Sites

Extract MODIS cloud gap-filled snow cover fraction (MOD10A1F) for the three GEM sites
(Zackenberg, Nuuk, Disko) to generate snow-free day time series and trend analyses.

## Requirements

    conda env create -f environment.yml
    conda activate modis_scf

## Pipeline

All steps are run from the project root. Use `make <target> SITE=<site>` or call scripts directly.

### 1. Download

    make download SITE=zackenberg
    # python scripts/download.py --site zackenberg

Downloads MOD10A1F HDF files from NASA Earthdata via `earthaccess`.
Requires Earthdata credentials in `~/.netrc`.
Raw HDF files go to `/mnt/glaciologi/GEM/remote_sensing_and_modelling/data/MODIS_CGF_SCF/`.

### 2. Clip & convert

    make clip SITE=zackenberg
    # python scripts/clip.py --site zackenberg

Reprojects to UTM (per-site EPSG), clips to AOI shapefile, writes annual NetCDFs to
`netcdf/{site}/`. Extracts `CGF_NDSI_Snow_Cover` and `Cloud_Persistence`.

### 3. Mask

    make mask SITE=zackenberg
    # python scripts/mask_scf.py --site zackenberg

Rasterises land/ice shapefiles onto the SCF grid, producing `snow_cover_fraction_masked`
(ice-free land only) in `netcdf/{site}_masked/`.

### 4. Terrain

    make terrain SITE=zackenberg
    # python scripts/get_terrain.py --site zackenberg

Extracts elevation, slope, and aspect from ArcticDEM and reprojects to the MODIS grid.
Output: `netcdf/{site}/{site}_terrain.nc`.

### 5. Snow-free days

    make snow-free-days SITE=zackenberg
    # python scripts/get_snow_free_days.py --site zackenberg

Computes mean snow-free days per year across NDSI thresholds 10–70%.
Outputs: `results/csvs/{site}/snow_free_days.csv`, figures in `figures/{site}/`.

### 6. Validate mast pixel

    make validate SITE=zackenberg
    # python scripts/validate_mast_pixel.py --site zackenberg

Compares MODIS mast-pixel snow-free days against in situ albedo and snow depth from the
GEM climate station. Outputs scatter plots and threshold performance curves in `figures/{site}/`.

### 7. Glacier SCF validation (supporting evidence)

    make validate-glacier SITE=zackenberg
    # python scripts/validate_glacier_scf.py --site zackenberg

Compares MODIS-observed SCF on glacier pixels (`masked_withice`) against RF-predicted SCF
(`filled`) by 100 m elevation band. Produces side-by-side line plots, heatmaps, and a
multi-year mean profile with ±1 std.

**Finding**: Raw MODIS NDSI on glacier pixels is not suitable for snow-free day detection.
Bare glacier ice produces NDSI values above the snow threshold (even at 60%), making it
indistinguishable from snow-covered surfaces. This confirms that glacier pixels must be
excluded from direct NDSI-based analysis and that RF gap-filling is the only viable
approach for extending SCF estimates to glacier areas.

### 8. Glacier gap-filling

    make fill-glaciers SITE=zackenberg
    # python scripts/fill_glacier_scf.py --site zackenberg

Trains a Random Forest on observed land pixels (features: elevation, slope, sin/cos aspect,
UTM easting, sin/cos DOY, year) and predicts SCF for glacier-masked pixels. Days with fewer
than 10% valid land observations (e.g. polar night) are left as NaN.
Output: `netcdf/{site}_filled/{site}_scf_{year}_filled.nc` with variables `scf_filled`
(land=observed, glacier=RF-predicted) and `fill_flag`.

### 8. Elevation band analysis

    make elevation-bands SITE=zackenberg
    # python scripts/analyse_elevation_bands.py --site zackenberg --source masked
    # python scripts/analyse_elevation_bands.py --site zackenberg --source filled

Bins pixels into 100 m elevation bands and computes mean annual snow-free days per band.
Running both sources generates comparison line plots and a difference heatmap automatically.

### 9. Climate predictors

    make derive-climate
    # python scripts/derive_climate_predictors.py

Derives PDD, winter precipitation, and melt days from CARRA reanalysis NetCDFs.
Output: `results/csvs/{site}/climate_predictors_{site}.csv`.

### 10. Analysis

    make analyse
    # python scripts/analyse_trends.py          — Mann-Kendall + Sen's slope
    # python scripts/analyse_driver_trends.py   — trends in PDD, winter precip, SFD
    # python scripts/analyse_correlations.py    — Pearson, partial correlations, VIF
    # python scripts/analyse_regression.py      — OLS regression with diagnostics

All analysis scripts write figures to `figures/{site}/` and LaTeX table snippets alongside them.

## Project Structure

    ├── environment.yml
    ├── Makefile
    ├── config/
    │   ├── zackenberg.yml               # Site config (EPSG, AOI, paths)
    │   ├── nuuk.yml
    │   └── disko.yml
    ├── scripts/
    │   ├── melt_model/                  # Degree-day melt model (separate from observations)
    │   │   └── compute_melt.py          # Gridded daily snow/ice melt from CARRA + SCF
    │   ├── download.py                  # Download MOD10A1F HDF files via earthaccess
    │   ├── clip.py                      # Reproject, clip to AOI, write annual NetCDFs
    │   ├── mask_scf.py                  # Apply land/ice masks (glacier pixels → NaN)
    │   ├── get_terrain.py               # Extract ArcticDEM elevation/slope/aspect
    │   ├── get_snow_free_days.py        # Compute snow-free days across NDSI thresholds
    │   ├── validate_mast_pixel.py       # Compare MODIS mast pixel vs in situ observations
    │   ├── validate_glacier_scf.py      # Glacier SCF: observed vs RF-predicted by elevation band
    │   ├── fill_glacier_scf.py          # RF glacier gap-filling + validation
    │   ├── analyse_elevation_bands.py   # Snow-free days by 100 m elevation band
    │   ├── derive_climate_predictors.py # Compute PDD, winter precip, melt days (CARRA)
    │   ├── load_climate_drivers.py      # Helper for loading CARRA climate data
    │   ├── analyse_trends.py            # Mann-Kendall trend test on snow-free days
    │   ├── analyse_driver_trends.py     # Trends in PDD, winter precip, and snow-free days
    │   ├── analyse_correlations.py      # Pearson and partial correlations + VIF
    │   └── analyse_regression.py        # OLS regression with residual diagnostics
    ├── data/
    │   └── CARRA/                       # CARRA reanalysis NetCDFs (t2m, tp at stations)
    ├── shp/                             # AOI polygons and station positions
    ├── netcdf/
    │   ├── {site}/                      # Annual SCF NetCDFs (clipped) + terrain
    │   ├── {site}_masked/               # Annual SCF NetCDFs (land only — ocean + glaciers masked)
    │   ├── {site}_masked_withice/       # Annual SCF NetCDFs (ocean masked, glaciers retained)
    │   └── {site}_filled/               # Annual SCF NetCDFs (glacier gap-filled by RF)
    ├── results/
    │   └── csvs/                        # Intermediate and final CSVs
    └── figures/
        └── {site}/                      # Output plots and LaTeX tables

## Melt Model

Scripts in `scripts/melt_model/` are kept separate from the observation pipeline.
They consume outputs from the observation pipeline (terrain, masked SCF, gap-filled SCF,
CARRA temperature) but represent a distinct modelling step.

    make melt SITE=zackenberg
    # python scripts/melt_model/compute_melt.py --site zackenberg

Applies a degree-day model to produce daily gridded snow and ice melt (mm w.e./day):
- CARRA 2m temperature is downscaled to each MODIS pixel via a standard lapse rate
- Surface type (snow / bare ice / bare land) is determined from masked and gap-filled SCF
- Melt = DDF × PDD, with separate degree-day factors for snow and ice
- Output: `netcdf/{site}_melt/{site}_melt_{year}.nc`

## Key Conventions

- NDSI threshold of **40%** is the primary snow-free threshold; sensitivity tested across 10–70%.
- MODIS flag values (>100) represent cloud/missing/water and are excluded from all analysis.
- Glacier gap-filling is suppressed on days with <10% valid land observations (polar night).
- Hydrological year runs **September to September** for winter precipitation.
- Statistical significance threshold: **α = 0.05** throughout.

## Notes

- Earthdata login credentials must be stored in `~/.netrc` for `download.py`.
- Raw HDF files are at `/mnt/glaciologi/GEM/remote_sensing_and_modelling/data/MODIS_CGF_SCF/`.
- Land/ice shapefiles for masking are from `/home/shl/mdrev/data/sdfi/G250_VEKTOR/`.
- In situ GEM station data is read from paths defined in `config/{site}.yml`.
- ArcticDEM tiles are mosaicked and read from the path in `config/{site}.yml`.
