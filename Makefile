PYTHON = /home/shl/miniconda3/envs/modis_scf/bin/python

# Run with: make <target> SITE=zackenberg
SITE ?= zackenberg

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

preprocess: clip mask

# ============================================================
# Analysis (Zackenberg only for now)
# ============================================================

derive-climate:
	$(PYTHON) scripts/derive_climate_predictors.py

trends:
	$(PYTHON) scripts/analyse_trends.py

driver-trends:
	$(PYTHON) scripts/analyse_driver_trends.py

correlations:
	$(PYTHON) scripts/analyse_correlations.py

regression:
	$(PYTHON) scripts/analyse_regression.py

analyse: trends driver-trends correlations regression

# ============================================================
# Convenience targets per site
# ============================================================

download-zackenberg:
	$(PYTHON) scripts/download.py --site zackenberg

download-disko:
	$(PYTHON) scripts/download.py --site disko

.PHONY: download clip mask snow-free-days preprocess \
        derive-climate trends driver-trends correlations regression analyse \
        download-zackenberg download-disko
