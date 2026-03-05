"""
Snow-Free Day Analysis for MODIS CGF NDSI Snow Cover
=====================================================
Computes snow-free days per year across multiple thresholds,
produces a threshold sensitivity time series plot and an
NDSI value distribution histogram.

Usage:
    python analyse_snow_free_days.py

Adjust nc_dir and aoi_path below to match your setup.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Config ---
nc_dir = Path("./netcdf/zackenberg_masked/")
out_dir = Path("./figures/zackenberg/")
csv_out_dir = Path("./csvs/zackenberg/")
out_dir.mkdir(parents=True, exist_ok=True)
csv_out_dir.mkdir(parents=True, exist_ok=True)

thresholds = [10, 20, 30, 40, 50, 60, 70]
variable = "snow_cover_fraction_masked"

# --- Load all annual NetCDFs into one dataset ---
nc_files = sorted(nc_dir.glob("zackenberg_scf_*.nc"))
if not nc_files:
    raise FileNotFoundError(f"No NetCDF files found in {nc_dir}")

print(f"Found {len(nc_files)} annual files")
ds = xr.open_mfdataset(nc_files, combine="by_coords")
da = ds[variable]

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
ax.set_title("Zackenberg: Snow-Free Days by NDSI Threshold")
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
latex_path = out_dir / "snow_free_days_summary.tex"
with open(latex_path, "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Mean snow-free days per year by NDSI threshold, "
            "Zackenberg.}\n")
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
