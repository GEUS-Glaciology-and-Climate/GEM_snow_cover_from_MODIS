PYTHON = /home/shl/miniconda3/envs/modis_scf/bin/python

# Run with: make <target> SITE=zackenberg
SITE ?= nuuk

# ============================================================
# Per-site pipeline
# ============================================================

download:
	$(PYTHON) scripts/download.py --site $(SITE)

clip:
	$(PYTHON) scripts/clip.py --site $(SITE)

mask:
	$(PYTHON) scripts/mask_scf.py --site $(SITE)

snow-free-days:
	$(PYTHON) scripts/get_snow_free_days.py --site $(SITE)

terrain:
	$(PYTHON) scripts/get_terrain.py --site $(SITE)

validate:
	$(PYTHON) scripts/validate_mast_pixel.py --site $(SITE)

validate-winter-precip:
	$(PYTHON) scripts/validate_winter_precip.py --site $(SITE)

validate-winter-precip-all:
	$(PYTHON) scripts/validate_winter_precip.py --site zackenberg
	$(PYTHON) scripts/validate_winter_precip.py --site nuuk
	$(PYTHON) scripts/validate_winter_precip.py --site disko

fill-glaciers:
	$(PYTHON) scripts/fill_glacier_scf.py --site $(SITE)

fill-glaciers-xgb:
	$(PYTHON) scripts/fill_glacier_scf_xgb.py --site $(SITE)

# Runs masked then filled so the comparison plots are generated automatically
elevation-bands:
	$(PYTHON) scripts/analyse_elevation_bands.py --site $(SITE) --source masked
	$(PYTHON) scripts/analyse_elevation_bands.py --site $(SITE) --source filled

preprocess: clip mask

pre-analysis: snow-free-days terrain validate fill-glaciers elevation-bands

# ============================================================
# Analysis
# ============================================================

derive-climate:
	$(PYTHON) scripts/derive_climate_predictors.py --site $(SITE)

trends:
	$(PYTHON) scripts/analyse_trends.py --site $(SITE)

driver-trends:
	$(PYTHON) scripts/analyse_driver_trends.py --site $(SITE)

correlations:
	$(PYTHON) scripts/analyse_correlations.py --site $(SITE)

regression:
	$(PYTHON) scripts/analyse_regression.py --site $(SITE)

analyse: derive-climate trends driver-trends correlations regression

.PHONY: download clip mask snow-free-days terrain validate validate-glacier fill-glaciers \
        fill-glaciers-xgb elevation-bands preprocess derive-climate trends driver-trends \
        correlations regression analyse validate-winter-precip validate-winter-precip-all
