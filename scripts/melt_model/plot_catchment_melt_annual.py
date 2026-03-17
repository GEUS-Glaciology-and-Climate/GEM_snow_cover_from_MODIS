"""
Per-Year Catchment Melt vs Discharge Plots
===========================================
One figure per year showing the seasonal cycle of:
  - Snow melt + ice melt (stacked area, left axis)
  - Observed river discharge (line, left axis)
  - CARRA precipitation (bar, right axis, inverted)

Outputs
-------
  figures/{site}/catchment_melt_annual/  — one PNG per year

Usage
-----
  python scripts/melt_model/plot_catchment_melt_annual.py --site zackenberg
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
gem_dir = Path(cfg["gem_dir"])
fig_dir = Path(f"figures/{site}/catchment_melt_annual/")
csv_dir = Path(f"results/csvs/{site}/")
fig_dir.mkdir(parents=True, exist_ok=True)

# ── load catchment daily melt ──────────────────────────────────────────────────
melt_csv = csv_dir / "catchment_melt_daily.csv"
if not melt_csv.exists():
    raise FileNotFoundError(f"{melt_csv} not found. Run catchment_melt.py first.")

df_melt = pd.read_csv(melt_csv, parse_dates=["date"], index_col="date")
df_melt.index = pd.DatetimeIndex(df_melt.index).normalize()

# ── load river discharge ───────────────────────────────────────────────────────
print("Loading river discharge...")
q_files = list(gem_dir.glob("Discharge at a cross section*_data.txt"))
q_raw   = pd.read_csv(q_files[0], sep="\t", na_values=-9999, encoding="utf-8-sig")
q_raw["datetime"] = pd.to_datetime(q_raw["Date"] + " " + q_raw["Time"])
q_daily = (q_raw.set_index("datetime")["Q (m3/s)"]
           .resample("1D").mean()
           .rename("discharge_m3s"))
q_daily.index = q_daily.index.normalize()

# ── load CARRA precipitation ───────────────────────────────────────────────────
print("Loading CARRA precipitation...")
ds_tp = xr.open_dataset("data/CARRA/CARRA_tp_nearest_PROMICE.nc",
                         decode_timedelta=False)

station_names = ds_tp["name"].values.tolist()
st_idx = station_names.index(site)

tp_raw = ds_tp["tp"].isel(station=st_idx)   # (time, step), mm (kg/m²)

# Unstack to a flat valid_time series, then resample to daily sum.
# Each (base_time, step) gives a valid_time = base_time + step hours.
# Steps 6 and 18 from 00Z and 12Z runs cover four 6-h windows per day.
# We keep unique valid_times (dropping duplicates by taking the mean) then
# sum over calendar day to get daily precipitation in mm.
base_times = pd.DatetimeIndex(ds_tp.time.values)
step_hours = ds_tp.step.values   # [6, 18]

records = []
for si, sh in enumerate(step_hours):
    valid_times = base_times + pd.to_timedelta(sh, unit="h")
    vals = tp_raw.isel(step=si).values
    records.append(pd.Series(vals, index=valid_times))

tp_series = pd.concat(records).sort_index()
# Drop duplicate valid_times (overlapping forecast windows) — keep first
tp_series = tp_series[~tp_series.index.duplicated(keep="first")]
# Resample to daily total (mm/day)
tp_daily = tp_series.resample("1D").sum()
tp_daily.index = tp_daily.index.normalize()

# Only keep rain: precipitation when daily mean T > 0°C
# (use CARRA t2m to separate rain from snow for the plot label)
ds_t2m = xr.open_dataset("data/CARRA/CARRA_t2m_nearest_PROMICE.nc",
                          decode_timedelta=False)
t2m_raw = ds_t2m["t2m"].isel(station=st_idx) - 273.15
t2m_daily = (t2m_raw.resample(time="1D").mean()
             .to_series()
             .rename("t2m"))
t2m_daily.index = t2m_daily.index.normalize()

rain_daily  = tp_daily.where(t2m_daily > 0, other=0.0)
snow_p_daily = tp_daily.where(t2m_daily <= 0, other=0.0)

# ── plot one figure per year ───────────────────────────────────────────────────
years = sorted(df_melt.index.year.unique())
# Exclude years with no melt data at all
years = [y for y in years if df_melt.loc[df_melt.index.year == y,
                                         "total_melt_mm"].sum() > 0]

print(f"Plotting {len(years)} years...")

for year in years:
    melt_yr = df_melt.loc[df_melt.index.year == year]
    q_yr    = q_daily.loc[q_daily.index.year == year]
    rain_yr = rain_daily.loc[rain_daily.index.year == year]
    t_yr    = t2m_daily.loc[t2m_daily.index.year == year]

    doy_melt = melt_yr.index.dayofyear
    doy_q    = q_yr.index.dayofyear
    doy_rain = rain_yr.index.dayofyear

    fig, ax1 = plt.subplots(figsize=(11, 4.5))

    # --- Left axis: melt + discharge ---
    ax1.fill_between(doy_melt, 0, melt_yr["snow_melt_mm"],
                     color="steelblue", alpha=0.7, label="Snow melt")
    ax1.fill_between(doy_melt, melt_yr["snow_melt_mm"],
                     melt_yr["snow_melt_mm"] + melt_yr["ice_melt_mm"],
                     color="firebrick", alpha=0.7, label="Ice melt")

    if q_yr.notna().sum() > 10:
        ax1.plot(doy_q, q_yr.values * 86400 / (399e6) * 1e3,
                 color="black", linewidth=1.5, label="Discharge (obs)")

    ax1.set_xlabel("Day of year", fontsize=9)
    ax1.set_ylabel("Runoff equivalent (mm day⁻¹)", fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.set_xlim(1, 366)
    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax1.grid(True, alpha=0.25)

    # --- Right axis: precipitation (inverted, bars) ---
    ax2 = ax1.twinx()
    ax2.bar(doy_rain, rain_yr.values,
            width=1, color="darkgreen", alpha=0.5, label="Rain (T > 0°C)")
    ax2.set_ylabel("Precipitation (mm day⁻¹)", fontsize=9, color="darkgreen")
    ax2.tick_params(axis="y", labelcolor="darkgreen", labelsize=8)
    ax2.set_ylim(bottom=0)
    # Invert so bars hang down from the top
    ax2.invert_yaxis()
    ymax = max(rain_yr.max() * 3, 5)
    ax2.set_ylim(ymax, 0)

    # --- Combined legend ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=8, loc="upper left", ncol=4)

    # Annual totals annotation
    snow_tot = melt_yr["snow_melt_mm"].sum()
    ice_tot  = melt_yr["ice_melt_mm"].sum()
    q_tot    = (q_yr * 86400 / (399e6) * 1e3).sum()
    ann_str  = (f"Snow: {snow_tot:.0f} mm  |  Ice: {ice_tot:.0f} mm  |  "
                f"Total melt: {snow_tot+ice_tot:.0f} mm")
    if q_yr.notna().sum() > 10:
        ann_str += f"  |  Discharge: {q_tot:.0f} mm"
    ax1.set_title(f"{site.capitalize()} {year}: catchment melt vs discharge\n"
                  f"{ann_str}", fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_dir / f"catchment_melt_{year}.png", dpi=150)
    plt.close(fig)
    print(f"  {year}: saved")

print(f"\nDone. {len(years)} plots saved to figures/{site}/catchment_melt_annual/")
