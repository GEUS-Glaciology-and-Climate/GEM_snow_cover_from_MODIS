"""
Glacier SCF Validation — Observed vs RF-predicted by elevation band
====================================================================
Compares MODIS-observed SCF on glacier pixels (from masked_withice)
against RF-predicted SCF (from filled) to assess how well the gap-
filling captures snow dynamics on glaciers.

For each 100 m elevation band:
  - Mean annual snow-free days from MODIS observed glacier pixels
  - Mean annual snow-free days from RF-predicted glacier pixels

A good match indicates the RF is reproducing observed glacier snow
dynamics; systematic divergence reveals model bias by elevation.

Outputs:
  figures/{site}/glacier_validation_sfd_bands_lines.png
  figures/{site}/glacier_validation_sfd_bands_heatmap.png
  figures/{site}/glacier_validation_sfd_scatter.png
  results/csvs/{site}/glacier_validation_sfd_bands.csv

Usage:
    python scripts/validate_glacier_scf.py --site zackenberg
"""

import argparse
import yaml
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

THRESHOLD_PRED = 40   # RF-predicted glacier SCF (trained on land pixels)
THRESHOLD_OBS  = 60   # Observed glacier SCF (higher threshold to account for bare ice)
BIN_WIDTH  = 100
MIN_PIXELS = 3

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True)
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site    = cfg["site"]
fig_dir = Path(f"figures/{site}/")
csv_dir = Path(f"results/csvs/{site}/")
fig_dir.mkdir(parents=True, exist_ok=True)
csv_dir.mkdir(parents=True, exist_ok=True)

# --- Load datasets ---
withice_dir = Path(f"netcdf/{site}_masked_withice/")
masked_dir  = Path(f"netcdf/{site}_masked/")
filled_dir  = Path(f"netcdf/{site}_filled/")

for d in [withice_dir, masked_dir, filled_dir]:
    if not d.exists():
        raise FileNotFoundError(f"Directory not found: {d}")

print("Loading withice SCF...")
ds_wi = xr.open_mfdataset(sorted(withice_dir.glob(f"{site}_scf_*.nc")),
                           combine="by_coords")
da_wi = ds_wi["snow_cover_fraction_masked"].where(
    ds_wi["snow_cover_fraction_masked"] <= 100)

print("Loading masked SCF (land only)...")
ds_m = xr.open_mfdataset(sorted(masked_dir.glob(f"{site}_scf_*.nc")),
                          combine="by_coords")
da_m = ds_m["snow_cover_fraction_masked"].where(
    ds_m["snow_cover_fraction_masked"] <= 100)

print("Loading filled SCF...")
ds_f = xr.open_mfdataset(sorted(filled_dir.glob(f"{site}_scf_*.nc")),
                          combine="by_coords")
da_filled = ds_f["scf_filled"]
da_flag   = ds_f["fill_flag"]

# --- Build glacier-only spatial mask ---
# Glacier pixels = valid in withice but NaN in masked (any time step)
# Use first time step that has broad coverage to identify glacier extent
ref_wi = da_wi.isel(time=100).values
ref_m  = da_m.isel(time=100).values
glacier_mask = np.isfinite(ref_wi) & ~np.isfinite(ref_m)

# Fallback: use flag from filled dataset (flag==1 marks glacier pixels)
if glacier_mask.sum() == 0:
    glacier_mask = (da_flag.isel(time=100).values == 1)

print(f"Glacier pixels identified: {glacier_mask.sum():,}")

# --- Load and reproject elevation to MODIS grid ---
terrain_path = Path(f"netcdf/{site}/{site}_terrain.nc")
terrain = xr.open_dataset(terrain_path)
da_tmpl = da_wi.isel(time=0).drop_vars("time").rio.write_crs(cfg["target_epsg"])
elev_modis = (terrain["elevation"]
              .rio.write_crs(cfg["target_epsg"])
              .rio.reproject_match(da_tmpl)
              .values)

elev_glacier = elev_modis[glacier_mask]
valid_elev   = elev_glacier[np.isfinite(elev_glacier)]
elev_min = np.floor(valid_elev.min() / BIN_WIDTH) * BIN_WIDTH
elev_max = np.ceil(valid_elev.max()  / BIN_WIDTH) * BIN_WIDTH
bin_edges  = np.arange(elev_min, elev_max + BIN_WIDTH, BIN_WIDTH)
bin_labels = bin_edges[:-1].astype(int)

print(f"Glacier elevation range: {elev_min:.0f} – {elev_max:.0f} m")
print(f"Number of {BIN_WIDTH} m bins: {len(bin_labels)}\n")

bin_idx_2d = np.full(elev_modis.shape, -1, dtype=int)
finite_mask = np.isfinite(elev_modis)
bin_idx_2d[finite_mask] = np.clip(
    np.digitize(elev_modis[finite_mask], bin_edges) - 1,
    0, len(bin_labels) - 1,
)

# --- Compute annual SFD per elevation band for both sources ---
years = np.unique(da_wi.time.dt.year.values)
records = []

for year in years:
    # Observed glacier SCF (withice, glacier pixels only)
    obs_yr = da_wi.sel(time=da_wi.time.dt.year == year)
    if len(obs_yr.time) < 200:
        print(f"  {year}: skipped (withice, {len(obs_yr.time)} days)")
        continue

    obs_np = obs_yr.values.astype(np.float32)  # (ndays, ny, nx)

    # RF-predicted glacier SCF (filled, glacier pixels only)
    pred_yr  = da_filled.sel(time=da_filled.time.dt.year == year)
    flag_yr  = da_flag.sel(time=da_flag.time.dt.year == year)
    pred_np  = pred_yr.values.astype(np.float32)
    flag_np  = flag_yr.values

    # Mask to glacier pixels only
    # obs: NaN on non-glacier pixels already (land-only masked), keep as-is for glacier
    obs_glacier  = np.where(glacier_mask[np.newaxis, :, :], obs_np,  np.nan)
    pred_glacier = np.where(flag_np == 1,                   pred_np, np.nan)

    # SFD maps: count days where SCF < threshold (NaN days not counted)
    sfd_obs  = np.nansum(obs_glacier  < THRESHOLD_OBS,  axis=0).astype(float)
    sfd_pred = np.nansum(pred_glacier < THRESHOLD_PRED, axis=0).astype(float)

    # Zero out pixels with no valid obs at all
    n_obs_days  = np.sum(np.isfinite(obs_glacier),  axis=0)
    n_pred_days = np.sum(np.isfinite(pred_glacier), axis=0)
    sfd_obs[n_obs_days   == 0] = np.nan
    sfd_pred[n_pred_days == 0] = np.nan

    for b, label in enumerate(bin_labels):
        mask_b = (bin_idx_2d == b) & glacier_mask
        n_pix  = mask_b.sum()
        if n_pix < MIN_PIXELS:
            continue

        obs_in_bin  = sfd_obs[mask_b]
        pred_in_bin = sfd_pred[mask_b]

        n_obs_valid  = np.sum(np.isfinite(obs_in_bin))
        n_pred_valid = np.sum(np.isfinite(pred_in_bin))

        records.append({
            "year":           year,
            "elev_lower":     int(label),
            "n_pixels":       int(n_pix),
            "sfd_obs_mean":   float(np.nanmean(obs_in_bin))  if n_obs_valid  >= MIN_PIXELS else np.nan,
            "sfd_obs_std":    float(np.nanstd(obs_in_bin))   if n_obs_valid  >= MIN_PIXELS else np.nan,
            "sfd_pred_mean":  float(np.nanmean(pred_in_bin)) if n_pred_valid >= MIN_PIXELS else np.nan,
            "sfd_pred_std":   float(np.nanstd(pred_in_bin))  if n_pred_valid >= MIN_PIXELS else np.nan,
            "n_obs_valid":    int(n_obs_valid),
            "n_pred_valid":   int(n_pred_valid),
        })

    print(f"  {year}: done")

df = pd.DataFrame(records)
csv_path = csv_dir / "glacier_validation_sfd_bands.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved CSV: {csv_path}")

# --- Pivot tables ---
piv_obs  = df.pivot(index="elev_lower", columns="year", values="sfd_obs_mean").sort_index()
piv_pred = df.pivot(index="elev_lower", columns="year", values="sfd_pred_mean").sort_index()

# ============================================================
# Plot 1: Line plots — observed vs predicted per year
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)

cmap2  = plt.cm.viridis
years_avail = sorted(set(piv_obs.columns) | set(piv_pred.columns))
colors = {yr: cmap2(i / max(len(years_avail) - 1, 1))
          for i, yr in enumerate(years_avail)}

for ax, piv, title in [
    (axes[0], piv_obs,  f"MODIS observed (glacier, NDSI < {THRESHOLD_OBS}%)"),
    (axes[1], piv_pred, f"RF predicted (glacier, NDSI < {THRESHOLD_PRED}%)"),
]:
    for year in piv.columns:
        col = piv[year].dropna()
        if len(col) < 2:
            continue
        ax.plot(col.values, col.index + BIN_WIDTH / 2,
                marker="o", markersize=2.5, linewidth=1.2, color=colors[year])
    ax.set_xlabel("Mean snow-free days per year", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

axes[0].set_ylabel("Elevation (m a.s.l.)", fontsize=9)

sm = plt.cm.ScalarMappable(cmap=cmap2,
     norm=plt.Normalize(vmin=min(years_avail), vmax=max(years_avail)))
sm.set_array([])
fig.colorbar(sm, ax=axes, label="Year", shrink=0.6, pad=0.02)
fig.suptitle(
    f"{site.capitalize()}: Glacier snow-free days by elevation — "
    f"observed (NDSI < {THRESHOLD_OBS}%) vs RF predicted (NDSI < {THRESHOLD_PRED}%)",
    fontsize=11)
fig.tight_layout()
fig.savefig(fig_dir / "glacier_validation_sfd_bands_lines.png", dpi=150)
plt.close(fig)
print("Saved line plot")

# ============================================================
# Plot 2: Heatmaps side by side
# ============================================================
common_years = piv_obs.columns.intersection(piv_pred.columns)
common_elev  = piv_obs.index.intersection(piv_pred.index)
vmax = max(np.nanmax(piv_obs.values), np.nanmax(piv_pred.values))

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
extent = [common_years.min() - 0.5, common_years.max() + 0.5,
          common_elev.min() - BIN_WIDTH / 2, common_elev.max() + BIN_WIDTH / 2]

for ax, piv, title in [
    (axes2[0], piv_obs.loc[common_elev, common_years],  f"MODIS observed (NDSI < {THRESHOLD_OBS}%)"),
    (axes2[1], piv_pred.loc[common_elev, common_years], f"RF predicted (NDSI < {THRESHOLD_PRED}%)"),
]:
    im = ax.imshow(piv.values, aspect="auto", origin="lower",
                   extent=extent, cmap="YlOrRd", vmin=0, vmax=vmax)
    fig2.colorbar(im, ax=ax, label="Mean SFD per year", shrink=0.8)
    ax.set_xlabel("Year"); ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

axes2[0].set_ylabel("Elevation (m a.s.l.)")
fig2.suptitle(
    f"{site.capitalize()}: Glacier SFD by elevation (NDSI < {THRESHOLD_OBS}% obs / {THRESHOLD_PRED}% pred)", fontsize=11)
fig2.tight_layout()
fig2.savefig(fig_dir / "glacier_validation_sfd_bands_heatmap.png", dpi=150)
plt.close(fig2)
print("Saved heatmap")

# ============================================================
# Plot 3: Mean profile across all years ± std, observed vs predicted
# ============================================================
obs_mean  = piv_obs.loc[common_elev].mean(axis=1)
obs_std   = piv_obs.loc[common_elev].std(axis=1)
pred_mean = piv_pred.loc[common_elev].mean(axis=1)
pred_std  = piv_pred.loc[common_elev].std(axis=1)
elev_centers = common_elev + BIN_WIDTH / 2

fig3, ax3 = plt.subplots(figsize=(6, 7))
ax3.fill_betweenx(elev_centers,
                  obs_mean - obs_std, obs_mean + obs_std,
                  alpha=0.2, color="steelblue")
ax3.fill_betweenx(elev_centers,
                  pred_mean - pred_std, pred_mean + pred_std,
                  alpha=0.2, color="firebrick")
ax3.plot(obs_mean,  elev_centers, "-o", markersize=4, linewidth=1.8,
         color="steelblue",  label=f"MODIS observed (NDSI < {THRESHOLD_OBS}%)")
ax3.plot(pred_mean, elev_centers, "-s", markersize=4, linewidth=1.8,
         color="firebrick", label=f"RF predicted (NDSI < {THRESHOLD_PRED}%)")
ax3.set_xlabel("Mean snow-free days per year", fontsize=9)
ax3.set_ylabel("Elevation (m a.s.l.)", fontsize=9)
ax3.set_title(
    f"{site.capitalize()}: Multi-year mean glacier SFD profile (±1 std)\n"
    f"Observed NDSI < {THRESHOLD_OBS}%  |  RF predicted NDSI < {THRESHOLD_PRED}%",
    fontsize=10)
ax3.yaxis.set_major_locator(ticker.MultipleLocator(200))
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(100))
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)
ax3.tick_params(labelsize=8)
fig3.tight_layout()
fig3.savefig(fig_dir / "glacier_validation_sfd_profile.png", dpi=150)
plt.close(fig3)
print("Saved mean profile plot")

print("\nDone.")
