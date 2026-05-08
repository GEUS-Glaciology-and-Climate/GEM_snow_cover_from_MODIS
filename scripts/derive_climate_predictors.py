"""
Derive Climate Predictors for Snow-Free Day Analysis 
=================================================================
Computes from CARRA reanalysis:
  1. PDD (Positive Degree Days) — calendar year sum of positive daily means
  2. Winter precipitation — hydrological year (Sept–Sept) total precip when T < 0°C
  3. Melt days — calendar year count of days with daily mean T > -3°C

Computes from in situ (GEM station):
  4. PDD from station data for comparison with CARRA

Outputs:
  - climate_predictors.csv: year, pdd_carra, winter_precip, melt_days
  - pdd_comparison.csv: year, pdd_carra, pdd_insitu
  - PDD comparison plot

Usage:
    python derive_climate_predictors.py --site zackenberg
"""

import argparse
import yaml
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Config ---

carra_tp_path = "./data/CARRA/CARRA_tp_nearest_PROMICE.nc"
carra_t2m_path = "./data/CARRA/CARRA_t2m_nearest_PROMICE.nc"

melt_threshold = -3.0  # °C, daily mean threshold for "melt day"

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True, help="Site name matching a config/<site>.yml")
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

station           = cfg["station_id"]
gem_dir           = cfg["gem_dir"]
air_temp          = cfg["insitu_temperature"]
insitu_path       = Path(f"{gem_dir}/{air_temp}")
site              = cfg["site"]
csv_dir           = Path(f"./results/csvs/{site}/")
fig_dir           = Path(f"./figures/{site}/")  
csv_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

# Column name overrides — gets the insitu column in  the config file if it exists, otherwise it falls back to the standard name 
col_temperature       = cfg.get("insitu_col_temperature",           "AT (°C)")

# ============================================================
# 1. Load CARRA temperature
# ============================================================
print("Loading CARRA temperature...")
ds_t2m = xr.load_dataset(carra_t2m_path)
i = int(np.where(ds_t2m["name"].values == station)[0][0])
t2m = ds_t2m["t2m"].isel(station=i) - 273.15  # K -> °C

# Daily mean temperature
t2m_daily = t2m.resample(time="1D").mean()

# ============================================================
# 2. PDD from CARRA (calendar year)
# ============================================================
print("Computing CARRA PDD...")
# Require >= 320 days of CARRA data per calendar year to avoid partial-year sums
carra_days_per_year = t2m_daily.notnull().resample(time="YE").sum()
valid_carra_years = carra_days_per_year["time"].dt.year.values[carra_days_per_year.values >= 320]

pdd_daily = t2m_daily.where(t2m_daily > 0, 0)
pdd_annual = pdd_daily.resample(time="YE").sum()

pdd_carra = pd.DataFrame({
    "year": pdd_annual["time"].dt.year.values,
    "pdd_carra": pdd_annual.values
}).set_index("year")
pdd_carra = pdd_carra[pdd_carra.index.isin(valid_carra_years)]

print(f"  CARRA PDD: {pdd_carra.index[0]}–{pdd_carra.index[-1]}")

# ============================================================
# 3. Melt days from CARRA (calendar year)
# ============================================================
print(f"Computing melt days (daily mean > {melt_threshold}°C)...")
melt_day_flag = t2m_daily > melt_threshold
melt_days_annual = melt_day_flag.resample(time="YE").sum()

melt_days = pd.DataFrame({
    "year": melt_days_annual["time"].dt.year.values,
    "melt_days": melt_days_annual.values.astype(int)
}).set_index("year")
melt_days = melt_days[melt_days.index.isin(valid_carra_years)]

print(f"  Melt days: {melt_days.index[0]}–{melt_days.index[-1]}")

# ============================================================
# 4. Winter precipitation from CARRA (hydrological year, Sept–Sept)
# ============================================================
print("Computing winter precipitation (hydro year, T < 0°C)...")
ds_tp = xr.load_dataset(carra_tp_path)
i_tp = int(np.where(ds_tp["name"].values == station)[0][0])
# CARRA tp is accumulated from forecast init: step=6h is 0-6h, step=18h is 0-18h.
# Take the difference to get the non-overlapping 6-18h increment per 12h cycle,
# giving two non-overlapping 12h blocks per day with no double-counting.
tp_raw = ds_tp["tp"].isel(station=i_tp)
tp = tp_raw.isel(step=1) - tp_raw.isel(step=0)

# Convert to mm if in metres
tp_units = ds_tp["tp"].attrs.get("units", "").lower()
if tp_units in ["m", "meter", "metre", "meters", "metres"]:
    tp = tp * 1000.0

# Align precip and temperature
tp, t2m_aligned = xr.align(tp, t2m, join="inner")

# Keep only precip when T < 0°C
tp_cold = tp.where(t2m_aligned < 0)

# Hydrological year totals (year ending in September)
wp_hydro = tp_cold.resample(time="YE-SEP").sum()

# The hydro year label: year ending Sept 2001 represents winter going into 2001 melt
winter_precip = pd.DataFrame({
    "year": wp_hydro["time"].dt.year.values,
    "winter_precip_mm": wp_hydro.values
}).set_index("year")

print(f"  Winter precip: {winter_precip.index[0]}–{winter_precip.index[-1]}")

# ============================================================
# 5. PDD from in situ (calendar year)
# ============================================================
print("Loading in situ temperature...")
insitu = pd.read_csv(insitu_path, sep="\t", na_values=-9999)
insitu["date"] = pd.to_datetime(insitu["Date"] + " " + insitu["Time"])
insitu.set_index("date", inplace=True)
insitu.rename(columns={col_temperature: "air_temperature_2m"}, inplace=True)

# Resample to daily mean
insitu_daily = insitu["air_temperature_2m"].resample("1D").mean()

# PDD: sum of positive daily means per calendar year
pdd_insitu_daily = insitu_daily.where(insitu_daily > 0, 0)
pdd_insitu_annual = pdd_insitu_daily.resample("YE").sum()

pdd_insitu = pd.DataFrame({
    "year": pdd_insitu_annual.index.year,
    "pdd_insitu": pdd_insitu_annual.values
}).set_index("year")

# Drop years with too much missing data (< 320 days)
days_per_year = insitu_daily.dropna().resample("YE").count()
valid_years = days_per_year[days_per_year >= 320].index.year
pdd_insitu = pdd_insitu.loc[pdd_insitu.index.isin(valid_years)]

print(f"  In situ PDD: {pdd_insitu.index[0]}–{pdd_insitu.index[-1]} "
      f"({len(pdd_insitu)} valid years)")

# ============================================================
# 6. Combine and export climate predictors
# ============================================================
print("\nCombining predictors...")
predictors = pdd_carra.join(winter_precip, how="outer").join(melt_days, how="outer")

# Save all predictors
pred_path = f"{csv_dir}/climate_predictors_{site}.csv"
predictors.to_csv(pred_path)
print(f"Saved: {pred_path}")
print(predictors.dropna().to_string())

# ============================================================
# 7. PDD comparison: CARRA vs in situ
# ============================================================
print("\nPDD comparison...")
pdd_comp = pdd_carra.join(pdd_insitu, how="inner")

comp_path = f"{csv_dir}/pdd_comparison_{site}.csv"
pdd_comp.to_csv(comp_path)
print(f"Saved: {comp_path}")

# --- Scatter plot ---
fig, ax = plt.subplots(figsize=(7, 7))

x = pdd_comp["pdd_insitu"].values
y = pdd_comp["pdd_carra"].values

ax.scatter(x, y, color="steelblue", s=40, zorder=3)

# 1:1 line
lims = [min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.1]
ax.plot(lims, lims, "--", color="gray", linewidth=1, label="1:1 line")

# Linear fit
coeffs = np.polyfit(x, y, 1)
r2 = np.corrcoef(x, y)[0, 1] ** 2
rmse = np.sqrt(np.mean((y - x) ** 2))
bias = np.mean(y - x)
fit_x = np.linspace(lims[0], lims[1], 100)
ax.plot(fit_x, np.polyval(coeffs, fit_x), "-", color="firebrick", linewidth=1.5,
        label=f"Fit: y={coeffs[0]:.2f}x + {coeffs[1]:.1f}")

ax.set_xlabel("In situ PDD (°C·days)")
ax.set_ylabel("CARRA PDD (°C·days)")
ax.set_title(f"{site}: PDD Comparison — CARRA vs In Situ")
ax.legend(loc="upper left")
ax.text(0.95, 0.05,
        f"R² = {r2:.3f}\nRMSE = {rmse:.1f} °C·d\nBias = {bias:.1f} °C·d",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(fig_dir / "pdd_comparison_carra_vs_insitu.png", dpi=150)
print(f"Saved: pdd_comparison_carra_vs_insitu.png")

# --- Time series comparison ---
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(pdd_comp.index, pdd_comp["pdd_carra"], "o-", color="firebrick",
         markersize=4, linewidth=1.5, label="CARRA")
ax2.plot(pdd_comp.index, pdd_comp["pdd_insitu"], "o-", color="steelblue",
         markersize=4, linewidth=1.5, label="In situ")
ax2.set_xlabel("Year")
ax2.set_ylabel("PDD (°C·days)")
ax2.set_title(f"{site}: Annual PDD — CARRA vs In Situ")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(fig_dir / "pdd_timeseries_carra_vs_insitu.png", dpi=150)
print(f"Saved: pdd_timeseries_carra_vs_insitu.png")

print("\nDone.")
