"""
Ice Melt Validation — Ablation Stakes and Continuous Surface Lowering
=====================================================================
Validates the degree-day model's ice melt estimates at two glacier
AWS sites (ZAC_L / APO_690_C at 694 m, ZAC_U / APO_920_C at 920 m)
using two independent observational datasets:

  1. Annual stake balance (stake_balance_upto_2024.csv)
     — total summer ice ablation in m ice per year
     — converted to mm w.e. using ice density 900 kg/m³

  2. Continuous surface lowering (ice_surface_height_zac_*.tsv)
     — hourly surface height measurements
     — resampled to daily means, differenced for daily melt rate
     — converted to mm w.e./day: Δheight × 1000 × 0.9

Model ice melt is extracted from the MODIS pixel containing each site.

Outputs
-------
  figures/{site}/melt_validation_annual_stakes.png  — scatter: modelled vs observed annual melt
  figures/{site}/melt_validation_daily_rates.png    — time series: daily rates at both sites
  figures/{site}/melt_validation_daily_scatter.png  — scatter: daily modelled vs observed
  results/csvs/{site}/melt_validation_annual.csv
  results/csvs/{site}/melt_validation_daily.csv

Usage
-----
  python scripts/melt_model/validate_ice_melt.py --site zackenberg
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from pyproj import Transformer
from sklearn.metrics import r2_score, mean_squared_error

ICE_DENSITY = 900.0   # kg/m³

# ── args & config ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True)
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site        = cfg["site"]
target_epsg = cfg["target_epsg"]

melt_dir = Path(f"netcdf/{site}_melt/")
fig_dir  = Path(f"figures/{site}/")
csv_dir  = Path(f"results/csvs/{site}/")
fig_dir.mkdir(parents=True, exist_ok=True)
csv_dir.mkdir(parents=True, exist_ok=True)

# ── site definitions ──────────────────────────────────────────────────────────
sites_info = {
    "ZAC_L": {"stake_id": "APO_690_C", "aws_file": "ice_surface_height_zac_l.tsv",
               "color": "steelblue"},
    "ZAC_U": {"stake_id": "APO_920_C", "aws_file": "ice_surface_height_zac_u.tsv",
               "color": "firebrick"},
}

# ── load site positions and convert to UTM ────────────────────────────────────
pos = pd.read_csv("data/positions.csv", skiprows=0)
pos.columns = pos.columns.str.strip()
pos.index = pos.iloc[:, 0].str.strip()
pos = pos.iloc[:, 1:]   # drop the name column, keep the rest

transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)

# ── get MODIS grid from first melt file ───────────────────────────────────────
melt_files = sorted(melt_dir.glob(f"{site}_melt_*.nc"))
if not melt_files:
    raise FileNotFoundError(f"No melt files in {melt_dir}. Run compute_melt.py first.")

ds_tmpl  = xr.open_dataset(melt_files[0])
x_coords = ds_tmpl.x.values
y_coords = ds_tmpl.y.values

def latlon_to_pixel(lat, lon):
    """Return (row, col) of the MODIS pixel containing (lat, lon)."""
    x_utm, y_utm = transformer.transform(lon, lat)
    col = int(np.argmin(np.abs(x_coords - x_utm)))
    row = int(np.argmin(np.abs(y_coords - y_utm)))
    return row, col

print("Site pixel locations:")
pixels = {}
for aws_name, info in sites_info.items():
    if aws_name not in pos.index:
        print(f"  {aws_name}: not found in positions.csv")
        continue
    lat = float(pos.loc[aws_name, "Latitude"])
    lon = float(pos.loc[aws_name, "Longitude"])
    row, col = latlon_to_pixel(lat, lon)
    pixels[aws_name] = (row, col)
    elev = float(pos.loc[aws_name, "Elevation"])
    print(f"  {aws_name}: lat={lat:.4f}, lon={lon:.4f} → pixel ({row},{col}), "
          f"elevation={elev:.0f} m")
ds_tmpl.close()

# ── extract modelled daily ice melt at each pixel ─────────────────────────────
print("\nExtracting modelled ice melt at stake pixels...")
model_daily = {aws: [] for aws in pixels}

for melt_file in melt_files:
    ds = xr.open_dataset(melt_file)
    times = pd.DatetimeIndex(ds.time.values)
    for aws_name, (row, col) in pixels.items():
        ice_melt = ds["ice_melt"].isel(y=row, x=col).values   # (ndays,) mm w.e./day
        for date, val in zip(times, ice_melt):
            model_daily[aws_name].append({"date": pd.Timestamp(date).normalize(),
                                          "ice_melt_model_mm": float(val)})
    ds.close()

model_dfs = {}
for aws_name, recs in model_daily.items():
    df = pd.DataFrame(recs).set_index("date").sort_index()
    df = df[~df.index.duplicated()]
    model_dfs[aws_name] = df
    print(f"  {aws_name}: {len(df)} days extracted")

# ── load and process stake annual balance ────────────────────────────────────
print("\nLoading annual stake balance...")
stakes = pd.read_csv("data/Stakes/stake_balance_upto_2024.csv")
stakes = stakes.set_index("Site Name")

annual_records = []
for aws_name, info in sites_info.items():
    stake_id = info["stake_id"]
    if stake_id not in stakes.index:
        print(f"  {stake_id}: not found in stake CSV")
        continue

    row_s = stakes.loc[stake_id]
    for col in row_s.index:
        if col in ("Elevation (arctic DEM)",):
            continue
        try:
            year = int(col)
        except ValueError:
            continue
        val = row_s[col]
        if pd.isna(val):
            continue

        # Convert m ice (negative = ablation) → mm w.e. (positive = melt)
        stake_mm = float(val) * -1 * 1000 * (ICE_DENSITY / 1000)

        # Sum modelled ice melt over the same year
        if aws_name not in model_dfs:
            continue
        model_yr = model_dfs[aws_name]
        model_yr_vals = model_yr.loc[model_yr.index.year == year, "ice_melt_model_mm"]
        model_mm = float(model_yr_vals.sum()) if len(model_yr_vals) > 0 else np.nan

        annual_records.append({
            "site":       aws_name,
            "stake_id":   stake_id,
            "year":       year,
            "stake_mm":   stake_mm,
            "model_mm":   model_mm,
        })

df_annual = pd.DataFrame(annual_records)
df_annual.to_csv(csv_dir / "melt_validation_annual.csv", index=False)
print(f"  Saved: {csv_dir}/melt_validation_annual.csv")
print(f"\n  Annual comparison (n={len(df_annual)}):")
print(f"  {'Site':>8}  {'Year':>6}  {'Stake (mm)':>11}  {'Model (mm)':>11}")
for _, r in df_annual.iterrows():
    print(f"  {r['site']:>8}  {r['year']:>6.0f}  {r['stake_mm']:>11.1f}  {r['model_mm']:>11.1f}")

# ── load and process continuous surface lowering ──────────────────────────────
print("\nLoading continuous surface height data...")
aws_daily = {}

for aws_name, info in sites_info.items():
    fpath = Path("data/AWS") / info["aws_file"]
    if not fpath.exists():
        print(f"  {aws_name}: file not found ({fpath})")
        continue

    df_raw = pd.read_csv(fpath, sep="\t", na_values=-9999, encoding="utf-8-sig")
    df_raw["datetime"] = pd.to_datetime(df_raw["Date"] + " " + df_raw["Time"])
    df_raw = df_raw.drop(columns=["Date", "Time"]).set_index("datetime")
    height_col = df_raw.columns[0]
    df_raw[height_col] = pd.to_numeric(df_raw[height_col], errors="coerce")

    # Resample to daily mean, then diff to get daily melt rate.
    # Increasing raw values = more melt, so positive diff = melt.
    daily_mean = df_raw[height_col].resample("1D").mean()
    daily_melt = daily_mean.diff()   # m/day, positive = ice loss

    # Convert to mm w.e./day: m × 1000 × (ice_density/water_density)
    daily_melt_mm = daily_melt * 1000 * (ICE_DENSITY / 1000)
    daily_melt_mm = daily_melt_mm.clip(lower=0)

    # Aggregate to weekly means — DDF is unreliable at daily timescales
    # Use min_count=3 so weeks with fewer than 3 valid days are set to NaN
    weekly_melt_mm = daily_melt_mm.resample("W").mean()

    aws_daily[aws_name] = weekly_melt_mm.rename(f"obs_melt_mm_{aws_name}")
    print(f"  {aws_name}: {daily_mean.first_valid_index().date()} to "
          f"{daily_mean.last_valid_index().date()}, "
          f"{weekly_melt_mm.notna().sum()} weeks")

# Aggregate modelled melt to weekly and merge with observed weekly rates
weekly_records = []
for aws_name, obs_series in aws_daily.items():
    if aws_name not in model_dfs:
        continue
    model_weekly = (model_dfs[aws_name]["ice_melt_model_mm"]
                    .resample("W").mean()
                    .rename("model_mm"))
    merged = (pd.DataFrame({"obs_mm": obs_series})
              .join(model_weekly, how="inner"))
    merged["site"] = aws_name
    weekly_records.append(merged)

df_weekly = pd.concat(weekly_records).reset_index().rename(columns={"index": "date"})
df_weekly.to_csv(csv_dir / "melt_validation_weekly.csv", index=False)
print(f"\n  Saved: {csv_dir}/melt_validation_weekly.csv")

# ── Figure 1: Annual stake scatter ────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))

for ax, aws_name in zip(axes1, sites_info):
    info = sites_info[aws_name]
    sub  = df_annual[df_annual["site"] == aws_name].dropna()
    if len(sub) < 2:
        ax.set_title(f"{aws_name}: insufficient data")
        continue

    lim = max(sub["stake_mm"].max(), sub["model_mm"].max()) * 1.1
    ax.scatter(sub["stake_mm"], sub["model_mm"],
               color=info["color"], s=60, zorder=3)
    ax.plot([0, lim], [0, lim], "--", color="gray", linewidth=1, label="1:1")

    for _, row in sub.iterrows():
        ax.annotate(str(int(row["year"])), (row["stake_mm"], row["model_mm"]),
                    fontsize=7, textcoords="offset points", xytext=(4, 2))

    r2   = r2_score(sub["stake_mm"], sub["model_mm"])
    bias = (sub["model_mm"] - sub["stake_mm"]).mean()
    rmse = float(np.sqrt(mean_squared_error(sub["stake_mm"], sub["model_mm"])))
    ax.text(0.05, 0.95,
            f"R² = {r2:.3f}\nRMSE = {rmse:.0f} mm\nBias = {bias:+.0f} mm",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    stake_id = info["stake_id"]
    elev = int(stakes.loc[stake_id, "Elevation (arctic DEM)"])
    ax.set_xlabel(f"Observed stake melt (mm w.e.)", fontsize=9)
    ax.set_ylabel("Modelled ice melt (mm w.e.)", fontsize=9)
    ax.set_title(f"{aws_name} / {stake_id} ({elev} m a.s.l.)", fontsize=10)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig1.suptitle(f"{site.capitalize()}: Annual ice melt — modelled vs stakes",
              fontsize=11)
fig1.tight_layout()
fig1.savefig(fig_dir / "melt_validation_annual_stakes.png", dpi=150)
plt.close(fig1)
print(f"Saved: figures/{site}/melt_validation_annual_stakes.png")

# ── Figure 2: Daily melt rate time series ─────────────────────────────────────
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

for ax, aws_name in zip(axes2, sites_info):
    info = sites_info[aws_name]
    if aws_name not in aws_daily or aws_name not in model_dfs:
        continue
    obs   = aws_daily[aws_name].dropna()
    model = model_dfs[aws_name]["ice_melt_model_mm"]

    obs_w   = aws_daily[aws_name].dropna()
    model_w = model_dfs[aws_name]["ice_melt_model_mm"].resample("W").mean()
    ax.fill_between(obs_w.index, 0, obs_w.values,
                    color=info["color"], alpha=0.5, label="Observed (weekly mean)")
    ax.plot(model_w.index, model_w.values,
            color="black", linewidth=0.8, alpha=0.8, label="Modelled ice melt (weekly mean)")
    ax.set_ylabel("Ice melt (mm w.e. day⁻¹)", fontsize=9)
    ax.set_title(f"{aws_name} / {info['stake_id']}", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    ax.set_ylim(bottom=0)

fig2.suptitle(f"{site.capitalize()}: Weekly mean ice melt rates — observed vs modelled",
              fontsize=11)
fig2.tight_layout()
fig2.savefig(fig_dir / "melt_validation_weekly_rates.png", dpi=150)
plt.close(fig2)
print(f"Saved: figures/{site}/melt_validation_weekly_rates.png")

# ── Figure 3: Weekly scatter at both sites ────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(10, 5))

for ax, aws_name in zip(axes3, sites_info):
    info = sites_info[aws_name]
    sub  = df_weekly[df_weekly["site"] == aws_name].dropna(subset=["obs_mm", "model_mm"])
    sub  = sub[(sub["obs_mm"] > 0) | (sub["model_mm"] > 0)]
    if len(sub) < 5:
        ax.set_title(f"{aws_name}: insufficient data")
        continue

    lim = max(sub["obs_mm"].quantile(0.99), sub["model_mm"].quantile(0.99)) * 1.1
    ax.scatter(sub["obs_mm"], sub["model_mm"],
               color=info["color"], alpha=0.5, s=30, rasterized=True)
    ax.plot([0, lim], [0, lim], "--", color="gray", linewidth=1, label="1:1")

    r2   = r2_score(sub["obs_mm"], sub["model_mm"])
    bias = (sub["model_mm"] - sub["obs_mm"]).mean()
    rmse = float(np.sqrt(mean_squared_error(sub["obs_mm"], sub["model_mm"])))
    ax.text(0.05, 0.95,
            f"R² = {r2:.3f}\nRMSE = {rmse:.1f} mm\nBias = {bias:+.1f} mm\nn = {len(sub)} weeks",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Observed melt (mm w.e. week⁻¹)", fontsize=9)
    ax.set_ylabel("Modelled melt (mm w.e. week⁻¹)", fontsize=9)
    ax.set_title(f"{aws_name} / {info['stake_id']}", fontsize=10)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig3.suptitle(f"{site.capitalize()}: Weekly ice melt — modelled vs observed (melt weeks only)",
              fontsize=11)
fig3.tight_layout()
fig3.savefig(fig_dir / "melt_validation_weekly_scatter.png", dpi=150)
plt.close(fig3)
print(f"Saved: figures/{site}/melt_validation_weekly_scatter.png")

print("\nDone.")
