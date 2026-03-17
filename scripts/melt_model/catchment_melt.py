"""
Catchment-Aggregated Melt vs River Discharge
=============================================
Masks the gridded melt outputs to a hydrological catchment shapefile,
sums daily snow and ice melt over the catchment, and compares against
observed river discharge.

Melt is reported in m³/day (volume) and as a runoff equivalent in mm/day
(volume / catchment area), matching the units of discharge.

Outputs
-------
  results/csvs/{site}/catchment_melt_daily.csv
  figures/{site}/catchment_melt_vs_discharge.png   — full time series
  figures/{site}/catchment_melt_seasonal.png        — mean seasonal cycle

Usage
-----
  python scripts/melt_model/catchment_melt.py --site zackenberg
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from pathlib import Path
from rasterio.features import rasterize
from rasterio.transform import Affine

# ── args & config ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True)
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site        = cfg["site"]
target_epsg = cfg["target_epsg"]
gem_dir     = Path(cfg["gem_dir"])

melt_dir = Path(f"netcdf/{site}_melt/")
fig_dir  = Path(f"figures/{site}/")
csv_dir  = Path(f"results/csvs/{site}/")
fig_dir.mkdir(parents=True, exist_ok=True)
csv_dir.mkdir(parents=True, exist_ok=True)

PIXEL_AREA_M2 = 500.0 * 500.0   # m² per MODIS pixel

# ── load catchment shapefile ──────────────────────────────────────────────────
catchment_shp = Path(f"shp/{site}_river_catchment_UTM.shp")
if not catchment_shp.exists():
    raise FileNotFoundError(f"Catchment shapefile not found: {catchment_shp}")

print("Loading catchment shapefile...")
catchment = gpd.read_file(catchment_shp).to_crs(epsg=target_epsg)
catchment_area_m2 = float(catchment.geometry.area.sum())
catchment_area_km2 = catchment_area_m2 / 1e6
print(f"  Catchment area: {catchment_area_km2:.1f} km²")

# ── rasterise catchment onto MODIS grid ───────────────────────────────────────
melt_files = sorted(melt_dir.glob(f"{site}_melt_*.nc"))
if not melt_files:
    raise FileNotFoundError(f"No melt files found in {melt_dir}. Run compute_melt.py first.")

ds_tmpl  = xr.open_dataset(melt_files[0])
x_coords = ds_tmpl.x.values
y_coords = ds_tmpl.y.values
ny, nx   = len(y_coords), len(x_coords)

res_x = float(x_coords[1] - x_coords[0])
res_y = float(y_coords[1] - y_coords[0])
transform = Affine(res_x, 0, float(x_coords[0]) - res_x / 2,
                   0, res_y, float(y_coords[0]) - res_y / 2)

shapes = [(geom, 1) for geom in catchment.geometry if geom is not None]
catchment_mask = rasterize(
    shapes, out_shape=(ny, nx), transform=transform,
    fill=0, dtype=np.uint8,
).astype(bool)

n_catchment_pixels = int(catchment_mask.sum())
catchment_area_modis_m2 = n_catchment_pixels * PIXEL_AREA_M2
print(f"  Catchment pixels (MODIS 500m): {n_catchment_pixels:,}")
print(f"  Catchment area from pixels:    {catchment_area_modis_m2/1e6:.1f} km²")
ds_tmpl.close()

# ── aggregate daily melt over catchment ───────────────────────────────────────
print("\nAggregating melt over catchment...")
records = []

for melt_file in melt_files:
    year = int(melt_file.stem.split("_")[-1])
    ds   = xr.open_dataset(melt_file)

    snow_np  = ds["snow_melt"].values    # (ndays, ny, nx) mm/day
    ice_np   = ds["ice_melt"].values
    total_np = ds["total_melt"].values
    times    = pd.DatetimeIndex(ds.time.values)

    for i, date in enumerate(times):
        # Sum over catchment pixels → mm × n_pixels → convert to m³/day
        snow_sum  = float(np.nansum(snow_np[i][catchment_mask]))
        ice_sum   = float(np.nansum(ice_np[i][catchment_mask]))
        total_sum = float(np.nansum(total_np[i][catchment_mask]))

        # m³/day = sum_mm × pixel_area_m² × 1e-3
        snow_m3  = snow_sum  * PIXEL_AREA_M2 * 1e-3
        ice_m3   = ice_sum   * PIXEL_AREA_M2 * 1e-3
        total_m3 = total_sum * PIXEL_AREA_M2 * 1e-3

        # Runoff equivalent mm/day = m³/day / catchment_area_m² × 1e3
        snow_mm  = snow_m3  / catchment_area_modis_m2 * 1e3
        ice_mm   = ice_m3   / catchment_area_modis_m2 * 1e3
        total_mm = total_m3 / catchment_area_modis_m2 * 1e3

        records.append({
            "date":         date,
            "snow_melt_m3": snow_m3,
            "ice_melt_m3":  ice_m3,
            "total_melt_m3": total_m3,
            "snow_melt_mm": snow_mm,
            "ice_melt_mm":  ice_mm,
            "total_melt_mm": total_mm,
        })

    print(f"  {year}: done")
    ds.close()

df = pd.DataFrame(records).set_index("date")
df.index = pd.DatetimeIndex(df.index)

csv_path = csv_dir / "catchment_melt_daily.csv"
df.to_csv(csv_path)
print(f"\nSaved: {csv_path}")

# ── load river discharge ───────────────────────────────────────────────────────
print("\nLoading river discharge...")
q_files = list(gem_dir.glob("Discharge at a cross section*_data.txt"))
if not q_files:
    raise FileNotFoundError(f"No discharge file found in {gem_dir}")

q_raw = pd.read_csv(
    q_files[0], sep="\t", na_values=-9999, encoding="utf-8-sig")
q_raw["datetime"] = pd.to_datetime(q_raw["Date"] + " " + q_raw["Time"])
q_raw = q_raw.set_index("datetime")

# Daily mean discharge (m³/s), then convert to m³/day
q_daily = q_raw["Q (m3/s)"].resample("1D").mean()
q_daily_m3 = q_daily * 86400          # m³/day
q_daily_mm = q_daily_m3 / catchment_area_modis_m2 * 1e3   # mm/day equivalent

print(f"  Discharge range: {q_daily.first_valid_index().date()} "
      f"to {q_daily.last_valid_index().date()}")
print(f"  Valid days: {q_daily.notna().sum()}")

# ── align to common period ────────────────────────────────────────────────────
common_idx = df.index.intersection(q_daily_mm.dropna().index)
print(f"  Overlapping days: {len(common_idx)}")

# ── figures ───────────────────────────────────────────────────────────────────
# Figure 1: Full time series — stacked melt + discharge
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top panel: melt components
ax = axes[0]
ax.fill_between(df.index, 0, df["snow_melt_mm"],
                color="steelblue", alpha=0.7, label="Snow melt")
ax.fill_between(df.index, df["snow_melt_mm"],
                df["snow_melt_mm"] + df["ice_melt_mm"],
                color="firebrick", alpha=0.7, label="Ice melt")
ax.set_ylabel("Melt (mm day⁻¹)", fontsize=9)
ax.set_title(f"{site.capitalize()}: Catchment melt — "
             f"snow + ice (DDF model, {catchment_area_modis_m2/1e6:.0f} km²)",
             fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=8)

# Bottom panel: discharge
ax2 = axes[1]
ax2.fill_between(q_daily_mm.index, 0, q_daily_mm,
                 color="teal", alpha=0.6, label="Observed discharge")
ax2.plot(df.index, df["total_melt_mm"], color="black",
         linewidth=0.8, alpha=0.7, label="Total melt (model)")
ax2.set_ylabel("Runoff equivalent (mm day⁻¹)", fontsize=9)
ax2.set_xlabel("Date", fontsize=9)
ax2.set_title("Observed discharge vs total modelled melt", fontsize=10)
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=8)
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

fig.tight_layout()
fig.savefig(fig_dir / "catchment_melt_vs_discharge.png", dpi=150)
plt.close(fig)
print(f"Saved: figures/{site}/catchment_melt_vs_discharge.png")

# Figure 2: Mean seasonal cycle (DOY) for overlapping years
df_common  = df.loc[common_idx]
q_common   = q_daily_mm.loc[common_idx]

df_common  = df_common.copy(); df_common["doy"] = df_common.index.dayofyear
q_common   = q_common.copy().to_frame("q_mm"); q_common["doy"] = q_common.index.dayofyear

melt_doy = df_common.groupby("doy")[["snow_melt_mm", "ice_melt_mm", "total_melt_mm"]].mean()
q_doy    = q_common.groupby("doy")["q_mm"].mean()

fig2, ax3 = plt.subplots(figsize=(10, 5))
ax3.fill_between(melt_doy.index, 0, melt_doy["snow_melt_mm"],
                 color="steelblue", alpha=0.7, label="Snow melt")
ax3.fill_between(melt_doy.index, melt_doy["snow_melt_mm"],
                 melt_doy["snow_melt_mm"] + melt_doy["ice_melt_mm"],
                 color="firebrick", alpha=0.7, label="Ice melt")
ax3.plot(q_doy.index, q_doy.values,
         color="teal", linewidth=2, label="Observed discharge")
ax3.set_xlabel("Day of year", fontsize=9)
ax3.set_ylabel("Runoff equivalent (mm day⁻¹)", fontsize=9)
ax3.set_title(f"{site.capitalize()}: Mean seasonal cycle — melt vs discharge\n"
              f"({common_idx.year.min()}–{common_idx.year.max()}, "
              f"overlapping years only)", fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(1, 365)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(30))
ax3.tick_params(labelsize=8)
fig2.tight_layout()
fig2.savefig(fig_dir / "catchment_melt_seasonal.png", dpi=150)
plt.close(fig2)
print(f"Saved: figures/{site}/catchment_melt_seasonal.png")

# ── summary stats ─────────────────────────────────────────────────────────────
print("\nAnnual totals over catchment (overlapping years):")
print(f"  {'Year':>6}  {'Snow (mm)':>10}  {'Ice (mm)':>10}  "
      f"{'Melt (mm)':>10}  {'Discharge (mm)':>14}")
print("  " + "-" * 58)
for year in sorted(common_idx.year.unique()):
    yr_m = df.loc[df.index.year == year]
    yr_q = q_daily_mm.loc[q_daily_mm.index.year == year]
    print(f"  {year:>6}  {yr_m['snow_melt_mm'].sum():>10.1f}  "
          f"{yr_m['ice_melt_mm'].sum():>10.1f}  "
          f"{yr_m['total_melt_mm'].sum():>10.1f}  "
          f"{yr_q.sum():>14.1f}")

print("\nDone.")
