"""
Snow-Free Days by Elevation Band
=================================
Reprojects ArcticDEM elevation to the MODIS 500m grid, bins pixels into
50 m elevation bands, and computes mean annual snow-free days per band.

Snow-free is defined as NDSI < 40% (calibrated threshold).

Outputs:
  results/csvs/{site}/sfd_elevation_bands.csv
  figures/{site}/sfd_elevation_bands_heatmap.png
  figures/{site}/sfd_elevation_bands_lines.png

Usage:
    python scripts/analyse_elevation_bands.py --site zackenberg
"""

import argparse
import yaml
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

THRESHOLD = 40
BIN_WIDTH = 100
MIN_PIXELS = 5   # minimum valid pixels per bin to include in output

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True)
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site    = cfg["site"]
nc_dir  = Path(f"netcdf/{site}_masked/")
out_dir = Path(f"figures/{site}/")
csv_dir = Path(f"results/csvs/{site}/")
out_dir.mkdir(parents=True, exist_ok=True)
csv_dir.mkdir(parents=True, exist_ok=True)

# --- Load SCF data ---
nc_files = sorted(nc_dir.glob(f"{site}_scf_*.nc"))
if not nc_files:
    raise FileNotFoundError(f"No masked NetCDF files found in {nc_dir}")

print(f"Loading {len(nc_files)} annual SCF files...")
ds = xr.open_mfdataset(nc_files, combine="by_coords")
da = ds["snow_cover_fraction_masked"].where(
    ds["snow_cover_fraction_masked"] <= 100
)

# --- Load terrain and reproject elevation to MODIS grid ---
terrain_path = Path(f"netcdf/{site}/{site}_terrain.nc")
if not terrain_path.exists():
    raise FileNotFoundError(f"Terrain file not found: {terrain_path}. Run make terrain first.")

print("Reprojecting elevation to MODIS grid...")
terrain = xr.open_dataset(terrain_path)
elev_32m = terrain["elevation"].rio.write_crs(cfg["target_epsg"])

# reproject_match reprojects and resamples to exactly the MODIS pixel grid
# averaging the 32m DEM pixels that fall within each 500m MODIS pixel
da_template = da.isel(time=0).drop_vars("time").rio.write_crs(cfg["target_epsg"])
elev_modis = elev_32m.rio.reproject_match(da_template)

elev_vals = elev_modis.values  # 2D array, NaN where ocean/no data

valid_elev = elev_vals[np.isfinite(elev_vals)]
elev_min = np.floor(valid_elev.min() / BIN_WIDTH) * BIN_WIDTH
elev_max = np.ceil(valid_elev.max()  / BIN_WIDTH) * BIN_WIDTH
bin_edges = np.arange(elev_min, elev_max + BIN_WIDTH, BIN_WIDTH)
bin_labels = bin_edges[:-1].astype(int)   # lower edge of each bin

print(f"Elevation range: {elev_min:.0f} – {elev_max:.0f} m")
print(f"Number of {BIN_WIDTH} m bins: {len(bin_labels)}")

# Assign each pixel to a bin index (-1 = outside range / NaN)
bin_idx = np.full(elev_vals.shape, -1, dtype=int)
valid_mask = np.isfinite(elev_vals)
bin_idx[valid_mask] = np.digitize(elev_vals[valid_mask], bin_edges) - 1
# Clip to valid range (top edge pixels land in bin == len(bin_labels))
bin_idx = np.clip(bin_idx, -1, len(bin_labels) - 1)

# --- Compute annual SFD per elevation bin ---
years = np.unique(da.time.dt.year.values)
records = []

for year in years:
    yearly = da.sel(time=da.time.dt.year == year)
    if len(yearly.time) < 320:
        print(f"  {year}: skipped ({len(yearly.time)} days)")
        continue

    sfd_map = (yearly < THRESHOLD).sum(dim="time").values.astype(float)
    # NaN where all days were invalid
    sfd_map[~np.isfinite(sfd_map)] = np.nan

    for b, label in enumerate(bin_labels):
        mask = bin_idx == b
        sfd_in_bin = sfd_map[mask]
        n_valid = np.sum(np.isfinite(sfd_in_bin))
        if n_valid < MIN_PIXELS:
            continue
        records.append({
            "year":       year,
            "elev_lower": label,
            "n_pixels":   int(n_valid),
            "sfd_mean":   float(np.nanmean(sfd_in_bin)),
            "sfd_std":    float(np.nanstd(sfd_in_bin)),
        })

    print(f"  {year}: done")

df = pd.DataFrame(records)

csv_path = csv_dir / "sfd_elevation_bands.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved CSV: {csv_path}")

# --- Pivot to year × elevation matrix for heatmap ---
pivot = df.pivot(index="elev_lower", columns="year", values="sfd_mean")
pivot = pivot.sort_index()   # ascending elevation top→bottom when plotted

# ============================================================
# Plot 1: Heatmap — elevation band × year, colour = SFD
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

im = ax.imshow(
    pivot.values,
    aspect="auto",
    origin="lower",
    extent=[
        pivot.columns.min() - 0.5,
        pivot.columns.max() + 0.5,
        pivot.index.min() - BIN_WIDTH / 2,
        pivot.index.max() + BIN_WIDTH / 2,
    ],
    cmap="YlOrRd",
    vmin=0,
)

cbar = fig.colorbar(im, ax=ax, label="Mean snow-free days per year")
ax.set_xlabel("Year")
ax.set_ylabel("Elevation (m a.s.l.)")
ax.set_title(f"{site.capitalize()}: Snow-free days by elevation band (NDSI < {THRESHOLD}%)")
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.tight_layout()
fig.savefig(out_dir / "sfd_elevation_bands_heatmap.png", dpi=150)
print("Saved heatmap")

# ============================================================
# Plot 2: Line plot — elevation on x, SFD on y, one line per year
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 7))

years_available = pivot.columns.tolist()
cmap2 = plt.cm.viridis
colors = cmap2(np.linspace(0.0, 1.0, len(years_available)))

for year, color in zip(years_available, colors):
    col = pivot[year].dropna()
    if len(col) < 3:
        continue
    elev_centers = col.index + BIN_WIDTH / 2
    ax2.plot(
        col.values, elev_centers,
        marker="o", markersize=2.5, linewidth=1.2,
        color=color,
    )

ax2.set_xlabel("Mean snow-free days per year")
ax2.set_ylabel("Elevation (m a.s.l.)")
ax2.set_title(f"{site.capitalize()}: Snow-free days by elevation band (NDSI < {THRESHOLD}%)")
ax2.yaxis.set_major_locator(ticker.MultipleLocator(200))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(50))
ax2.grid(True, alpha=0.3)

# Colorbar as year legend
sm = plt.cm.ScalarMappable(
    cmap=cmap2,
    norm=plt.Normalize(vmin=min(years_available), vmax=max(years_available)),
)
sm.set_array([])
fig2.colorbar(sm, ax=ax2, label="Year")

fig2.tight_layout()
fig2.savefig(out_dir / "sfd_elevation_bands_lines.png", dpi=150)
print("Saved line plot")
