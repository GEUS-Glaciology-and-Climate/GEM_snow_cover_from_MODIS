"""
Glacier SCF Gap-Filling via Random Forest
==========================================
Trains a Random Forest regressor on land pixels (observed SCF + terrain
features) and predicts SCF for glacier-masked pixels, producing daily
gap-filled snow cover maps.

Features
--------
  elevation, slope, sin(aspect), cos(aspect),
  sin(2π·DOY/365), cos(2π·DOY/365), year

Validation
----------
  1. Random spatial holdout  — 20% of land pixels withheld (all their days)
  2. Elevation holdout       — top 25% of land pixels by elevation withheld
     (most relevant: highest land pixels are most similar to glacier terrain)

Outputs
-------
  netcdf/{site}_filled/{site}_scf_{year}_filled.nc  — one per year
  figures/{site}/ml_validation_scatter.png
  figures/{site}/ml_validation_rmse_by_elevation.png
  results/csvs/{site}/ml_validation_stats.csv

Usage
-----
  python scripts/fill_glacier_scf.py --site zackenberg
"""

import argparse
import yaml
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from rasterio.features import rasterize as rio_rasterize

# ── constants ─────────────────────────────────────────────────────────────────
MAX_PER_DAY      = 200      # max land pixels sampled per day (keeps memory bounded)
MAX_TRAIN_SAMPLES = 500_000  # cap for RF training after accumulation
HOLDOUT_FRAC     = 0.20     # fraction of land pixels for random holdout
MIN_LAND_OBS_FRAC = 0.10    # min fraction of land pixels with valid obs to allow glacier fill
RF_PARAMS = dict(
    n_estimators     = 200,
    max_depth        = 20,
    min_samples_leaf = 5,
    n_jobs           = -1,
    random_state     = 42,
)
FEAT_NAMES = ["elevation", "slope", "sin_aspect", "cos_aspect",
              "easting", "sin_doy", "cos_doy", "year"]

# ── args & config ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True)
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site        = cfg["site"]
target_epsg = cfg["target_epsg"]
land_shp    = cfg["land_shp"]
ice_shp     = cfg["ice_shp"]

masked_dir = Path(f"netcdf/{site}_masked/")
out_dir    = Path(f"netcdf/{site}_filled/")
fig_dir    = Path(f"figures/{site}/")
csv_dir    = Path(f"results/csvs/{site}/")
for d in [out_dir, fig_dir, csv_dir]:
    d.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

# ── load one template to get grid geometry ────────────────────────────────────
nc_files = sorted(masked_dir.glob(f"{site}_scf_*.nc"))
if not nc_files:
    raise FileNotFoundError(f"No masked NetCDF files found in {masked_dir}")

ds_tmpl = xr.open_dataset(nc_files[0])
da_tmpl = (ds_tmpl["snow_cover_fraction_masked"]
           .isel(time=0).drop_vars("time")
           .rio.write_crs(target_epsg))
ny, nx  = da_tmpl.rio.height, da_tmpl.rio.width
transform = da_tmpl.rio.transform()
print(f"MODIS grid: {ny} × {nx} = {ny*nx:,} pixels")

# ── rasterize land and ice masks ──────────────────────────────────────────────
print("Rasterizing land and ice masks to MODIS grid...")
land_gdf = gpd.read_file(land_shp).to_crs(epsg=target_epsg)
ice_gdf  = gpd.read_file(ice_shp).to_crs(epsg=target_epsg)

land_mask = rio_rasterize(
    land_gdf.geometry, out_shape=(ny, nx), transform=transform,
    fill=0, default_value=1, dtype="uint8",
).astype(bool)
ice_mask = rio_rasterize(
    ice_gdf.geometry, out_shape=(ny, nx), transform=transform,
    fill=0, default_value=1, dtype="uint8",
).astype(bool)

land_only_mask = land_mask & ~ice_mask   # training pixels
land_flat = land_only_mask.ravel()
ice_flat  = ice_mask.ravel()
land_indices = np.where(land_flat)[0]    # flat indices of land pixels

print(f"  Land pixels (training targets): {land_only_mask.sum():,}")
print(f"  Glacier pixels (prediction targets): {ice_mask.sum():,}")

# ── load terrain, resample to MODIS grid ─────────────────────────────────────
terrain_path = Path(f"netcdf/{site}/{site}_terrain.nc")
if not terrain_path.exists():
    raise FileNotFoundError(f"Terrain not found: {terrain_path}. Run make terrain first.")

print("Loading terrain features...")
terrain = xr.open_dataset(terrain_path)
elev_modis  = terrain["elevation"].rio.write_crs(target_epsg).rio.reproject_match(da_tmpl).values
slope_modis = terrain["slope"].rio.write_crs(target_epsg).rio.reproject_match(da_tmpl).values
asp_modis   = terrain["aspect"].rio.write_crs(target_epsg).rio.reproject_match(da_tmpl).values

sin_asp = np.sin(np.radians(asp_modis))
cos_asp = np.cos(np.radians(asp_modis))

# 2D easting array (UTM x) — proxy for distance to coast / longitude gradient
x_coords = da_tmpl.x.values                          # 1D, length nx
east_modis = np.broadcast_to(x_coords, (ny, nx)).copy()

# Flatten terrain for fast indexing
elev_f  = elev_modis.ravel().astype(np.float32)
slope_f = slope_modis.ravel().astype(np.float32)
sin_f   = sin_asp.ravel().astype(np.float32)
cos_f   = cos_asp.ravel().astype(np.float32)
east_f  = east_modis.ravel().astype(np.float32)

# Report elevation overlap: land vs glacier
land_elev = elev_f[land_flat]
ice_elev  = elev_f[ice_flat]
print(f"\nElevation summary:")
print(f"  Land    : {np.nanmin(land_elev):.0f} – {np.nanmax(land_elev):.0f} m "
      f"(mean {np.nanmean(land_elev):.0f} m)")
print(f"  Glacier : {np.nanmin(ice_elev):.0f} – {np.nanmax(ice_elev):.0f} m "
      f"(mean {np.nanmean(ice_elev):.0f} m)")
extrapolation_pct = 100 * np.mean(ice_elev > np.nanmax(land_elev))
print(f"  Glacier pixels above land max: {extrapolation_pct:.1f}% (extrapolation risk)")

# ── build training matrix (year by year to limit memory) ─────────────────────
print(f"\nBuilding training matrix (max {MAX_PER_DAY} land pixels/day)...")
X_list, y_list, pid_list = [], [], []

for nc_file in sorted(nc_files):
    ds_yr = xr.open_dataset(nc_file)
    scf_yr = (ds_yr["snow_cover_fraction_masked"]
              .where(ds_yr["snow_cover_fraction_masked"] <= 100)
              .values.astype(np.float32))   # (ndays, ny, nx)
    times_yr = pd.DatetimeIndex(ds_yr.time.values)
    doys_yr  = times_yr.dayofyear.values
    years_yr = times_yr.year.values
    year_tag = years_yr[0]

    scf_flat = scf_yr.reshape(len(times_yr), -1)  # (ndays, npix)

    n_samples_yr = 0
    for i, (doy, year) in enumerate(zip(doys_yr, years_yr)):
        valid = land_flat & np.isfinite(scf_flat[i])
        n_valid = valid.sum()
        if n_valid == 0:
            continue

        # Subsample if too many valid land pixels
        idx = np.where(valid)[0]
        if n_valid > MAX_PER_DAY:
            idx = rng.choice(idx, MAX_PER_DAY, replace=False)

        sin_doy = np.sin(2 * np.pi * doy / 365)
        cos_doy = np.cos(2 * np.pi * doy / 365)
        n = len(idx)

        X_list.append(np.column_stack([
            elev_f[idx], slope_f[idx], sin_f[idx], cos_f[idx],
            east_f[idx],
            np.full(n, sin_doy, dtype=np.float32),
            np.full(n, cos_doy, dtype=np.float32),
            np.full(n, float(year), dtype=np.float32),
        ]))
        y_list.append(scf_flat[i, idx])
        pid_list.append(idx)
        n_samples_yr += n

    print(f"  {year_tag}: {n_samples_yr:,} samples")
    ds_yr.close()

X_all  = np.vstack(X_list).astype(np.float32)
y_all  = np.concatenate(y_list).astype(np.float32)
pid_all = np.concatenate(pid_list)
print(f"\nTotal training samples: {len(y_all):,}")

# ── holdout splits (on pixels, so all days of a pixel are in or out) ──────────
# 1. Random 20% of land pixels
n_holdout = int(len(land_indices) * HOLDOUT_FRAC)
holdout_rand = set(rng.choice(land_indices, n_holdout, replace=False))

# 2. Top-25% elevation land pixels (closest to glacier terrain)
elev_q75 = np.nanpercentile(land_elev, 75)
holdout_elev = set(land_indices[land_elev >= elev_q75])

def split(holdout_set):
    mask = np.array([p in holdout_set for p in pid_all])
    return ~mask, mask

train_r, test_r = split(holdout_rand)
train_e, test_e = split(holdout_elev)

# ── train, validate, report ───────────────────────────────────────────────────
def subsample_for_rf(X, y, max_n):
    if len(y) > max_n:
        idx = rng.choice(len(y), max_n, replace=False)
        return X[idx], y[idx]
    return X, y

def train_validate(X_tr, y_tr, X_te, y_te, label):
    X_s, y_s = subsample_for_rf(X_tr, y_tr, MAX_TRAIN_SAMPLES)
    print(f"\n[{label}] Training on {len(y_s):,} samples, testing on {len(y_te):,}...")
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_s, y_s)
    pred = rf.predict(X_te).clip(0, 100)
    r2   = r2_score(y_te, pred)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    bias = float(np.mean(pred - y_te))
    print(f"  R²={r2:.3f}  RMSE={rmse:.1f}  Bias={bias:+.2f}")
    return rf, pred, r2, rmse, bias

rf_r, pred_r, r2_r, rmse_r, bias_r = train_validate(
    X_all[train_r], y_all[train_r], X_all[test_r], y_all[test_r], "Random holdout")
rf_e, pred_e, r2_e, rmse_e, bias_e = train_validate(
    X_all[train_e], y_all[train_e], X_all[test_e], y_all[test_e], "Elevation holdout")

# ── save validation stats ─────────────────────────────────────────────────────
val_df = pd.DataFrame([
    {"holdout": "random_20pct",    "n_test": test_r.sum(),
     "r2": r2_r, "rmse": rmse_r, "bias": bias_r},
    {"holdout": "elevation_top25", "n_test": test_e.sum(),
     "r2": r2_e, "rmse": rmse_e, "bias": bias_e,
     "elev_threshold_m": float(elev_q75)},
])
val_df.to_csv(csv_dir / "ml_validation_stats.csv", index=False)
print(f"\nSaved: {csv_dir}/ml_validation_stats.csv")

# ── validation figures ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, (y_obs, y_pred, r2, rmse, bias, title) in zip(axes, [
    (y_all[test_r], pred_r, r2_r, rmse_r, bias_r,
     f"Random holdout ({int(HOLDOUT_FRAC*100)}% of land pixels)"),
    (y_all[test_e], pred_e, r2_e, rmse_e, bias_e,
     f"Elevation holdout (top 25%, ≥{elev_q75:.0f} m)"),
]):
    n_plot = min(30_000, len(y_obs))
    idx_p  = rng.choice(len(y_obs), n_plot, replace=False)
    elev_p = elev_f[pid_all[np.where(test_r if "Random" in title else test_e)[0][idx_p]]]
    sc = ax.scatter(y_obs[idx_p], y_pred[idx_p], c=elev_p, cmap="RdYlBu_r",
                    alpha=0.3, s=1, rasterized=True, vmin=0, vmax=1500)
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Observed SCF (%)")
    ax.set_ylabel("Predicted SCF (%)")
    ax.set_title(title)
    ax.text(0.05, 0.93, f"R²={r2:.3f}  RMSE={rmse:.1f}  Bias={bias:+.2f}",
            transform=ax.transAxes, fontsize=9)
    plt.colorbar(sc, ax=ax, label="Elevation (m)")

fig.suptitle(f"{site.capitalize()}: Random Forest validation", fontsize=12)
fig.tight_layout()
fig.savefig(fig_dir / "ml_validation_scatter.png", dpi=150)
print("Saved scatter plot")

# RMSE by elevation band for the elevation holdout (most informative)
elev_te = elev_f[pid_all[test_e]]
bin_edges = np.arange(
    np.floor(np.nanmin(elev_te) / 100) * 100,
    np.ceil(np.nanmax(elev_te)  / 100) * 100 + 100, 100)
rmse_bins, centers, n_bins = [], [], []
for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    m = (elev_te >= lo) & (elev_te < hi)
    if m.sum() < 20:
        continue
    rmse_bins.append(float(np.sqrt(mean_squared_error(y_all[test_e][m], pred_e[m]))))
    centers.append((lo + hi) / 2)
    n_bins.append(int(m.sum()))

fig2, ax2 = plt.subplots(figsize=(7, 6))
bars = ax2.barh(centers, rmse_bins, height=80, color="steelblue", alpha=0.8)
for bar, n in zip(bars, n_bins):
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
             f"n={n}", va="center", fontsize=7)
ax2.axvline(rmse_e, color="red", linestyle="--", lw=1, label=f"Overall RMSE={rmse_e:.1f}")
ax2.set_xlabel("RMSE (SCF %)")
ax2.set_ylabel("Elevation (m a.s.l.)")
ax2.set_title(f"{site.capitalize()}: Validation RMSE by elevation\n(elevation holdout — hardest test)")
ax2.yaxis.set_major_locator(ticker.MultipleLocator(200))
ax2.grid(True, alpha=0.3, axis="x")
ax2.legend()
fig2.tight_layout()
fig2.savefig(fig_dir / "ml_validation_rmse_by_elevation.png", dpi=150)
print("Saved RMSE-by-elevation plot")

# ── final model: train on all data ────────────────────────────────────────────
print("\nTraining final model on all land pixels...")
X_final, y_final = subsample_for_rf(X_all, y_all, MAX_TRAIN_SAMPLES)
rf_final = RandomForestRegressor(**RF_PARAMS)
rf_final.fit(X_final, y_final)

imp = pd.Series(rf_final.feature_importances_, index=FEAT_NAMES).sort_values(ascending=False)
print("Feature importances:")
for name, v in imp.items():
    print(f"  {name:<14}: {v:.3f}")

# ── predict glacier pixels and write filled NetCDFs ───────────────────────────
print("\nWriting filled NetCDFs...")

# Static terrain features for all glacier pixels (reused each day)
ice_idx   = np.where(ice_flat)[0]
n_ice     = len(ice_idx)
X_ice_static = np.column_stack([
    elev_f[ice_idx], slope_f[ice_idx], sin_f[ice_idx], cos_f[ice_idx],
    east_f[ice_idx],
]).astype(np.float32)  # (n_ice, 5) — DOY and year columns added per day below

for nc_file in sorted(nc_files):
    ds_yr = xr.open_dataset(nc_file)
    da_yr = (ds_yr["snow_cover_fraction_masked"]
             .where(ds_yr["snow_cover_fraction_masked"] <= 100))
    times_yr = pd.DatetimeIndex(da_yr.time.values)
    doys_yr  = times_yr.dayofyear.values
    years_yr = times_yr.year.values
    year_val = int(years_yr[0])

    scf_np = da_yr.values.astype(np.float32).copy()  # (ndays, ny, nx)
    flag_np = np.zeros(scf_np.shape, dtype=np.int8)
    scf_flat_yr = scf_np.reshape(len(times_yr), -1)

    n_land_total = int(land_flat.sum())
    min_land_obs = int(MIN_LAND_OBS_FRAC * n_land_total)
    n_skipped = 0

    for i, (doy, year) in enumerate(zip(doys_yr, years_yr)):
        # Skip glacier fill if too few land observations (e.g. polar night)
        n_valid_land = int(np.isfinite(scf_flat_yr[i][land_flat]).sum())
        if n_valid_land < min_land_obs:
            n_skipped += 1
            continue

        sin_doy = np.sin(2 * np.pi * doy / 365)
        cos_doy = np.cos(2 * np.pi * doy / 365)

        X_ice = np.hstack([
            X_ice_static,
            np.full((n_ice, 1), sin_doy, dtype=np.float32),
            np.full((n_ice, 1), cos_doy, dtype=np.float32),
            np.full((n_ice, 1), float(year), dtype=np.float32),
        ])

        pred = rf_final.predict(X_ice).clip(0, 100).astype(np.float32)

        flat = scf_np[i].ravel()
        flat[ice_idx] = pred
        scf_np[i] = flat.reshape(ny, nx)

        flat_flag = flag_np[i].ravel()
        flat_flag[ice_idx] = 1
        flag_np[i] = flat_flag.reshape(ny, nx)

    print(f"  {year_val}: {n_skipped} days skipped (< {MIN_LAND_OBS_FRAC*100:.0f}% land obs)")

    ds_out = xr.Dataset(
        {
            "scf_filled": (["time", "y", "x"], scf_np),
            "fill_flag":  (["time", "y", "x"], flag_np),
        },
        coords={"time": da_yr.time, "y": da_yr.y, "x": da_yr.x},
    )
    ds_out["scf_filled"].attrs = {
        "units":     "%",
        "long_name": "Snow cover fraction — land=observed, glacier=RF predicted",
    }
    ds_out["fill_flag"].attrs = {
        "long_name": "Fill source",
        "flag_values": "0, 1",
        "flag_meanings": "observed RF_predicted",
    }
    ds_out.attrs["source"]             = f"MODIS MOD10A1F + RF glacier infill; site={site}"
    ds_out.attrs["rf_r2_random"]       = f"{r2_r:.3f}"
    ds_out.attrs["rf_r2_elev"]         = f"{r2_e:.3f}"
    ds_out.attrs["rf_rmse_elev"]       = f"{rmse_e:.2f} %"
    ds_out.attrs["min_land_obs_frac"]  = f"{MIN_LAND_OBS_FRAC} (glacier fill suppressed below this fraction of valid land pixels)"

    out_path = out_dir / f"{site}_scf_{year_val}_filled.nc"
    ds_out.to_netcdf(out_path)
    print(f"  Wrote {out_path.name}")
    ds_yr.close()

print("\nDone.")
