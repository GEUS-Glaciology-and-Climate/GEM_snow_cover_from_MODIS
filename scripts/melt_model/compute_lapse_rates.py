"""
Observed Monthly Lapse Rates from AWS Temperatures
===================================================
Derives observed air temperature lapse rates from three elevation points:
  - CARRA 2m temperature at model orography (~105 m)
  - ZAC_L AWS air temperature (694 m)
  - ZAC_U AWS air temperature (920 m)

For each day with at least two valid observations, fits a linear regression
of temperature vs elevation. Monthly median lapse rates (°C/km) are then
computed and saved for use in the melt model.

Outputs
-------
  results/csvs/{site}/lapse_rates_monthly.csv  — 12 rows, one per month
  figures/{site}/lapse_rates_monthly.png        — boxplot of daily lapse rates by month

Usage
-----
  python scripts/melt_model/compute_lapse_rates.py --site zackenberg
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── args & config ─────────────────────────────────────────────────────────────
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

# Station elevations (m) — use CARRA model orography for CARRA
ELEV = {
    "carra": 105.0,
    "zac_l": 694.0,
    "zac_u": 920.0,
}

# ── helper: load AWS temperature tsv ─────────────────────────────────────────
def load_aws_temp(path):
    df = pd.read_csv(path, sep="\t", na_values=-9999, encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.set_index("datetime").drop(columns=["Date", "Time"])
    col = df.columns[0]
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[col].resample("1D").mean()

# ── load temperatures and resample to daily ───────────────────────────────────
print("Loading temperatures...")

# CARRA — 3-hourly in Kelvin
ds_carra = xr.open_dataset("data/CARRA/CARRA_t2m_nearest_PROMICE.nc",
                            decode_timedelta=False)
station_names = ds_carra["name"].values.tolist()
st_idx = station_names.index(site)
t_carra = ((ds_carra["t2m"].isel(station=st_idx) - 273.15)
           .resample(time="1D").mean()
           .to_series()
           .rename("carra"))
t_carra.index = t_carra.index.normalize()

# AWS stations
t_zac_l = load_aws_temp("data/AWS/air_temperature_zac_l.tsv").rename("zac_l")
t_zac_u = load_aws_temp("data/AWS/air_temperature_zac_u.tsv").rename("zac_u")

print(f"  CARRA:  {t_carra.first_valid_index().date()} – {t_carra.last_valid_index().date()} "
      f"({t_carra.notna().sum()} days)")
print(f"  ZAC_L:  {t_zac_l.first_valid_index().date()} – {t_zac_l.last_valid_index().date()} "
      f"({t_zac_l.notna().sum()} days)")
print(f"  ZAC_U:  {t_zac_u.first_valid_index().date()} – {t_zac_u.last_valid_index().date()} "
      f"({t_zac_u.notna().sum()} days)")

# ── merge into one dataframe ───────────────────────────────────────────────────
df = pd.concat([t_carra, t_zac_l, t_zac_u], axis=1)
df.index = pd.DatetimeIndex(df.index)

# ── compute daily lapse rate by linear regression through available points ────
print("\nComputing daily lapse rates...")
elevs = np.array([ELEV["carra"], ELEV["zac_l"], ELEV["zac_u"]])

lapse_rates = []
for date, row in df.iterrows():
    temps = row.values.astype(float)
    valid = np.isfinite(temps)
    if valid.sum() < 2:
        lapse_rates.append(np.nan)
        continue
    # Linear regression: T = a + b*elev → b is lapse rate in °C/m
    coeffs = np.polyfit(elevs[valid], temps[valid], 1)
    # Convert to °C/km (negative = temperature decreasing with elevation)
    lapse_rates.append(coeffs[0] * 1000)

df["lapse_rate_C_per_km"] = lapse_rates
print(f"  Valid lapse rate days: {df['lapse_rate_C_per_km'].notna().sum()}")
print(f"  Overall median: {df['lapse_rate_C_per_km'].median():.2f} °C/km")
print(f"  Range (5–95th pct): {df['lapse_rate_C_per_km'].quantile(0.05):.2f} to "
      f"{df['lapse_rate_C_per_km'].quantile(0.95):.2f} °C/km")

# ── monthly median lapse rates ────────────────────────────────────────────────
df["month"] = df.index.month
monthly = (df.groupby("month")["lapse_rate_C_per_km"]
           .agg(["median", "mean", "std",
                 lambda x: x.quantile(0.25),
                 lambda x: x.quantile(0.75),
                 "count"])
           .rename(columns={"median": "lapse_rate_median",
                            "mean":   "lapse_rate_mean",
                            "std":    "lapse_rate_std",
                            "<lambda_0>": "q25",
                            "<lambda_1>": "q75",
                            "count":  "n_days"}))

month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
monthly.index = [month_names[m] for m in monthly.index]
monthly.index.name = "month"

csv_path = csv_dir / "lapse_rates_monthly.csv"
monthly.to_csv(csv_path)
print(f"\nSaved: {csv_path}")
print(f"\n{'Month':>5}  {'Median (°C/km)':>15}  {'Mean':>8}  {'Std':>6}  {'N':>6}")
print("-" * 48)
for month, row in monthly.iterrows():
    print(f"{month:>5}  {row['lapse_rate_median']:>15.2f}  "
          f"{row['lapse_rate_mean']:>8.2f}  {row['lapse_rate_std']:>6.2f}  "
          f"{int(row['n_days']):>6}")

# ── figure: boxplot of daily lapse rates by month ────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

month_nums = sorted(df["month"].dropna().unique().astype(int))
data_by_month = [df.loc[df["month"] == m, "lapse_rate_C_per_km"].dropna().values
                 for m in month_nums]

bp = ax.boxplot(data_by_month, positions=month_nums, widths=0.6,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="firebrick", linewidth=2),
                boxprops=dict(facecolor="steelblue", alpha=0.5))

# Overlay monthly medians as connected line
medians = [np.median(d) for d in data_by_month if len(d) > 0]
ax.plot(month_nums, medians, "o-", color="firebrick",
        linewidth=1.5, markersize=5, zorder=5, label="Monthly median")

# Standard environmental lapse rate for reference
ax.axhline(-6.5, color="gray", linestyle="--", linewidth=1,
           label="Standard lapse rate (−6.5 °C/km)")
ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

ax.set_xlabel("Month", fontsize=9)
ax.set_ylabel("Lapse rate (°C/km)", fontsize=9)
ax.set_title(f"{site.capitalize()}: Observed monthly air temperature lapse rates\n"
             f"(CARRA 105 m – ZAC_L 694 m – ZAC_U 920 m, daily regression)",
             fontsize=10)
ax.set_xticks(month_nums)
ax.set_xticklabels([month_names[m] for m in month_nums], fontsize=8)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(fig_dir / "lapse_rates_monthly.png", dpi=150)
plt.close(fig)
print(f"\nSaved: figures/{site}/lapse_rates_monthly.png")
print("\nDone.")
