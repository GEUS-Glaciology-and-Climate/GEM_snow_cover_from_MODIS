"""
Snow-Free Day Analysis for MODIS CGF NDSI Snow Cover
=====================================================
Computes snow-free days per year across multiple thresholds,
produces a threshold sensitivity time series plot and an
NDSI value distribution histogram.

Also computes annual cloud persistence statistics (mean, std, min, max)
as a data quality indicator for the gap-filling.

Usage:
    python scripts/get_snow_free_days.py --site zackenberg

"""

import argparse
import yaml
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from pathlib import Path

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True, help="Site name matching a config/<site>.yml")
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site        = cfg["site"]
target_epsg = cfg["target_epsg"]
station     = cfg["station_id"]

nc_dir        = Path(f"netcdf/{site}_masked/")
out_dir       = Path(f"figures/{site}/")
csv_out_dir   = Path(f"results/csvs/{site}/")
latex_dir     = Path("results/latex/")
aws_stations_path = Path(cfg["aws_stations"])

out_dir.mkdir(parents=True, exist_ok=True)
csv_out_dir.mkdir(parents=True, exist_ok=True)
latex_dir.mkdir(parents=True, exist_ok=True)

thresholds = [10, 20, 30, 40, 50, 60, 70]
variable = "snow_cover_fraction_masked"

# --- Load all annual NetCDFs into one dataset ---
nc_files = sorted(nc_dir.glob(f"{site}_scf_*.nc"))
if not nc_files:
    raise FileNotFoundError(f"No NetCDF files found in {nc_dir}")

print(f"Found {len(nc_files)} annual files")
ds = xr.open_mfdataset(nc_files, combine="by_coords")
da = ds[variable]
cp = ds["cloud_persistence_masked"].where(ds["cloud_persistence_masked"] < 255)  # 255 = fill value

# Filter out flag values — only keep valid NDSI range 0-100
# Flags: 200=missing, 201=no decision, 211=night, 237=inland water,
# 239=ocean, 250=cloud, 254=detector saturated, 255=fill
da = da.where(da <= 100)

print(f"Dataset shape: {da.shape}")
print(f"Time range: {da.time.values[0]} to {da.time.values[-1]}")
print(f"Value range: {float(da.min()):.1f} to {float(da.max()):.1f}")

# ============================================================
# 1. Snow-free days time series for multiple thresholds
# ============================================================
# For each threshold: a pixel-day is "snow-free" if NDSI < threshold.
# Count per year per pixel, then average spatially.

years = np.unique(da.time.dt.year.values)
results = {t: [] for t in thresholds}
cp_stats = {"mean": [], "std": [], "min": [], "max": []}
years_used = []

for year in years:
    yearly = da.sel(time=da.time.dt.year == year)
    if len(yearly.time) < 320:
        print(f"  {year}: skipped ({len(yearly.time)} days)")
        continue
    years_used.append(year)
    for thresh in thresholds:
        snow_free = (yearly < thresh).sum(dim="time")  # per-pixel count
        mean_days = float(snow_free.mean())             # spatial average
        results[thresh].append(mean_days)

    yearly_cp = cp.sel(time=cp.time.dt.year == year)
    cp_stats["mean"].append(float(yearly_cp.mean()))
    cp_stats["std"].append(float(yearly_cp.std()))
    cp_stats["min"].append(float(yearly_cp.min()))
    cp_stats["max"].append(float(yearly_cp.max()))
    print(f"  {year}: done ({len(yearly.time)} days)")

# --- Plot: threshold sensitivity ---
fig, ax = plt.subplots(figsize=(12, 6))

cmap = plt.cm.viridis
colors = cmap(np.linspace(0.1, 0.9, len(thresholds)))

for thresh, color in zip(thresholds, colors):
    ax.plot(years_used, results[thresh], marker="o", markersize=3,
            label=f"< {thresh}%", color=color, linewidth=1.5)

ax.set_xlabel("Year")
ax.set_ylabel("Mean snow-free days per year")
ax.set_title("{site.capitalize()}: Snow-Free Days by NDSI Threshold")
ax.legend(title="Threshold", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xlim(years_used[0], years_used[-1])
fig.tight_layout()
fig.savefig(out_dir / "snow_free_days_threshold_sensitivity.png", dpi=150)
print(f"\nSaved threshold sensitivity plot")

# --- Export results to CSV ---
import pandas as pd

df = pd.DataFrame({"year": years_used})
for thresh in thresholds:
    df[f"sfd_lt{thresh}"] = results[thresh]

csv_path = csv_out_dir / "snow_free_days.csv"
df.to_csv(csv_path, index=False)
print(f"Saved CSV: {csv_path}")

# --- Export cloud persistence stats ---
cp_df = pd.DataFrame({
    "year": years_used,
    "cp_mean": cp_stats["mean"],
    "cp_std":  cp_stats["std"],
    "cp_min":  cp_stats["min"],
    "cp_max":  cp_stats["max"],
})
cp_csv_path = csv_out_dir / "cloud_persistence_stats.csv"
cp_df.to_csv(cp_csv_path, index=False)
print(f"Saved CSV: {cp_csv_path}")

print("\n" + "=" * 60)
print("Cloud persistence stats (days of gap-filling, 0 = no filling)")
print("=" * 60)
print(f"{'Year':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 42)
for _, row in cp_df.iterrows():
    print(f"{int(row.year):>6} {row.cp_mean:8.2f} {row.cp_std:8.2f} "
          f"{row.cp_min:8.0f} {row.cp_max:8.0f}")


# ============================================================
# 3. Print summary statistics
# ============================================================
print("\n" + "=" * 60)
print("Summary: Mean snow-free days per year by threshold")
print("=" * 60)
print(f"{'Threshold':>10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 42)
for thresh in thresholds:
    vals = np.array(results[thresh])
    print(f"{'< ' + str(thresh) + '%':>10} {vals.mean():8.1f} {vals.std():8.1f} "
          f"{vals.min():8.1f} {vals.max():8.1f}")


# --- Export summary as LaTeX table ---
latex_path = latex_dir / "snow_free_days_summary.tex"
with open(latex_path, "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Mean snow-free days per year by NDSI threshold, "
            f"{site.capitalize()}.}}\n")
    f.write("\\label{tab:sfd_summary}\n")
    f.write("\\begin{tabular}{lrrrr}\n")
    f.write("\\hline\n")
    f.write("Threshold & Mean & Std & Min & Max \\\\\n")
    f.write("\\hline\n")
    for thresh in thresholds:
        vals = np.array(results[thresh])
        f.write(f"$<$ {thresh}\\% & {vals.mean():.1f} & {vals.std():.1f} & "
                f"{vals.min():.1f} & {vals.max():.1f} \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"\nSaved LaTeX table: {latex_path}")

# --- Export cloud persistence stats as LaTeX table ---
cp_latex_path = latex_dir / "cloud_persistence_stats.tex"
with open(cp_latex_path, "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Annual cloud persistence statistics (ice-free land pixels), "
            f"{site.capitalize()}. Values indicate the number of days of gap-filling applied; "
            "0 = directly observed.}\n")
    f.write("\\label{tab:cloud_persistence}\n")
    f.write("\\begin{tabular}{lrrr}\n")
    f.write("\\hline\n")
    f.write("Year & Mean (days) & Std (days) & Max (days) \\\\\n")
    f.write("\\hline\n")
    for _, row in cp_df.iterrows():
        f.write(f"{int(row.year)} & {row.cp_mean:.2f} & {row.cp_std:.2f} & "
                f"{row.cp_max:.0f} \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"Saved LaTeX table: {cp_latex_path}")

# ============================================================
# 4. Supplementary overview: daily mean SCF and cloud persistence per year
# ============================================================
print("\nGenerating supplementary overview plot...")

ncols = 4
nrows = int(np.ceil(len(years_used) / ncols))
fig_ov, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3),
                             sharey=False)
axes = np.array(axes).flatten()

color_scf = "steelblue"
color_cp  = "darkorange"

for i, year in enumerate(years_used):
    ax = axes[i]
    ax2 = ax.twinx()

    scf_daily = da.sel(time=da.time.dt.year == year).mean(dim=["x", "y"])
    cp_daily  = cp.sel(time=cp.time.dt.year == year).mean(dim=["x", "y"])

    doy_scf = scf_daily.time.dt.dayofyear.values
    doy_cp  = cp_daily.time.dt.dayofyear.values

    ax.plot(doy_scf, scf_daily.values, color=color_scf, linewidth=1)
    ax2.plot(doy_cp, cp_daily.values, color=color_cp, linewidth=1, linestyle="--")

    ax.set_title(str(year), fontsize=9)
    ax.set_xlim(1, 366)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelcolor=color_scf, labelsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax2.tick_params(axis="y", labelcolor=color_cp, labelsize=7)
    ax2.set_ylim(bottom=0)

# Hide unused subplots
for j in range(len(years_used), len(axes)):
    axes[j].set_visible(False)

# Shared axis labels via figure text
fig_ov.text(0.5, 0.01, "Day of year", ha="center", fontsize=10)
fig_ov.text(0.01, 0.5, "Mean SCF (%)", va="center", rotation="vertical",
            color=color_scf, fontsize=10)
fig_ov.text(0.99, 0.5, "Mean cloud persistence (days)", va="center",
            rotation="vertical", color=color_cp, fontsize=10)

fig_ov.suptitle(f"{site.capitalize()}: Daily mean SCF and cloud persistence per year",
                fontsize=12, y=1.01)
fig_ov.tight_layout()
fig_ov.savefig(out_dir / "scf_cloud_persistence_overview.png", dpi=150,
               bbox_inches="tight")
print(f"Saved supplementary overview plot")

# ============================================================
# 5. Mast pixel extraction: snow/no-snow time series at AWS location
# ============================================================
print("\nExtracting mast pixel time series...")

# Load station coordinates and reproject to MODIS grid CRS
aws = pd.read_csv(aws_stations_path)
row = aws[aws["stid"] == station].iloc[0]
gdf = gpd.GeoDataFrame([row], geometry=gpd.points_from_xy([row.lon], [row.lat]),
                        crs="EPSG:4326").to_crs(epsg=target_epsg)
mast_x = float(gdf.geometry.x.iloc[0])
mast_y = float(gdf.geometry.y.iloc[0])
print(f"  {station} mast position: x={mast_x:.1f}, y={mast_y:.1f} (EPSG:{target_epsg})")

# Extract the single nearest pixel across the full time series
pixel = da.sel(x=mast_x, y=mast_y, method="nearest")
print(f"  Nearest MODIS pixel: x={float(pixel.x):.1f}, y={float(pixel.y):.1f}")

# Build output dataframe: date, raw SCF value, snow flag per threshold
mast_df = pd.DataFrame({"date": pixel.time.values, "scf": pixel.values})
for thresh in thresholds:
    mast_df[f"snow_lt{thresh}"] = (mast_df["scf"] < thresh).astype("Int64")
    # Set to NA where SCF was invalid (NaN)
    mast_df.loc[mast_df["scf"].isna(), f"snow_lt{thresh}"] = pd.NA

mast_csv_path = csv_out_dir / f"mast_pixel_snow_{station}.csv"
mast_df.to_csv(mast_csv_path, index=False)
print(f"  Saved: {mast_csv_path}")
