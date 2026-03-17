"""
Degree-Day Melt Model — Gridded Daily Snow and Ice Melt
========================================================
Downscales CARRA 2m air temperature to the MODIS 500 m grid using a
standard elevation lapse rate, then applies degree-day factors to
compute daily snow and ice melt for each pixel.

Surface classification per pixel per day
-----------------------------------------
  Ocean / outside AOI  → NaN
  Land (non-glacier)
    SCF >= SNOW_THRESHOLD   → snow melt  (DDF_snow × PDD)
    SCF <  SNOW_THRESHOLD   → no melt    (bare ground)
    SCF = NaN (cloud)       → NaN
  Glacier
    scf_filled >= SNOW_THRESHOLD → snow melt  (DDF_snow × PDD)
    scf_filled <  SNOW_THRESHOLD → ice  melt  (DDF_ice  × PDD)
    scf_filled = NaN (polar night) → NaN      (PDD ≈ 0 anyway)

PDD per pixel = max(T_pixel, 0)
T_pixel = T_carra − LAPSE_RATE(month) × (elev_pixel − elev_station) / 1000

Parameters
----------
  LAPSE_RATE    : observed monthly median lapse rates (°C/km) derived from
                  CARRA + ZAC_L + ZAC_U temperatures (compute_lapse_rates.py).
                  Falls back to 6.5 °C/km if the CSV is not found.
  DDF_snow      : 4.0 mm w.e. / °C / day
  DDF_ice       : 8.0 mm w.e. / °C / day
  SNOW_THRESHOLD: 40 % NDSI

Outputs
-------
  netcdf/{site}_melt/{site}_melt_{year}.nc   — one file per year
    snow_melt    (time, y, x)  mm w.e. day⁻¹
    ice_melt     (time, y, x)  mm w.e. day⁻¹
    total_melt   (time, y, x)  mm w.e. day⁻¹
    surface_type (time, y, x)  flag:
                               0 = ocean / no data
                               1 = snow-covered (land or glacier)
                               2 = bare ice
                               3 = bare land (no melt)
                               -1 = missing SCF (cloud)
  figures/{site}/melt_annual_totals.png
  results/csvs/{site}/melt_annual_totals.csv

Usage
-----
  python scripts/melt_model/compute_melt.py --site zackenberg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yaml

# ── parameters ────────────────────────────────────────────────────────────────
LAPSE_RATE_DEFAULT = 6.5    # °C per 1000 m — fallback if monthly CSV not found
DDF_SNOW           = 3.0    # mm w.e. / °C / day
DDF_ICE            = 7.0    # mm w.e. / °C / day
SNOW_THRESHOLD     = 40     # % NDSI

# surface_type flag values
FLAG_NODATA    = np.int8(0)
FLAG_SNOW      = np.int8(1)
FLAG_ICE       = np.int8(2)
FLAG_BARE_LAND = np.int8(3)
FLAG_CLOUD     = np.int8(-1)

# ── args & config ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True)
parser.add_argument("--ddf-snow", type=float, default=DDF_SNOW,
                    help=f"Degree-day factor for snow (default {DDF_SNOW})")
parser.add_argument("--ddf-ice",  type=float, default=DDF_ICE,
                    help=f"Degree-day factor for ice (default {DDF_ICE})")
parser.add_argument("--lapse-rate", type=float, default=None,
                    help="Override lapse rate (°C/km) for all months. "
                         "If not set, uses observed monthly lapse rates from CSV.")
args = parser.parse_args()

DDF_SNOW = args.ddf_snow
DDF_ICE  = args.ddf_ice

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site        = cfg["site"]
target_epsg = cfg["target_epsg"]

masked_dir = Path(f"netcdf/{site}_masked/")
filled_dir = Path(f"netcdf/{site}_filled/")
out_dir    = Path(f"netcdf/{site}_melt/")
fig_dir    = Path(f"figures/{site}/")
csv_dir    = Path(f"results/csvs/{site}/")
for d in [out_dir, fig_dir, csv_dir]:
    d.mkdir(parents=True, exist_ok=True)

# ── load monthly lapse rates ──────────────────────────────────────────────────
lapse_csv = Path(f"results/csvs/{site}/lapse_rates_monthly.csv")
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

if args.lapse_rate is not None:
    # User override: constant lapse rate for all months
    lapse_by_month = {m: args.lapse_rate for m in range(1, 13)}
    print(f"Using constant lapse rate (override): {args.lapse_rate} °C/km")
elif lapse_csv.exists():
    lr_df = pd.read_csv(lapse_csv, index_col=0)
    # Use observed lapse rates for melt season only (May–Sep).
    # Outside the melt season the lapse rate has negligible impact on melt
    # (PDD ≈ 0 everywhere), but shallow observed rates (inversions) would
    # generate spurious melt at high elevations on days when the valley
    # station is marginally above 0°C. Use standard rate for those months.
    MELT_SEASON = {5, 6, 7, 8, 9}
    lapse_by_month = {}
    for i, m in enumerate(month_names):
        month_num = i + 1
        if month_num in MELT_SEASON:
            # CSV stores negative values (°C/km, decreasing with elev).
            # Formula uses positive convention (negated internally), so negate.
            lapse_by_month[month_num] = -float(lr_df.loc[m, "lapse_rate_median"])
        else:
            lapse_by_month[month_num] = LAPSE_RATE_DEFAULT
    print("Using observed lapse rates (May–Sep) / standard (Oct–Apr) in °C/km:")
    for m, lr in lapse_by_month.items():
        src = "obs" if m in MELT_SEASON else "std"
        print(f"  {month_names[m-1]:>3}: {lr:+.2f}  ({src})")
else:
    lapse_by_month = {m: LAPSE_RATE_DEFAULT for m in range(1, 13)}
    print(f"Monthly lapse rate CSV not found — using default {LAPSE_RATE_DEFAULT} °C/km")
    print(f"  Run compute_lapse_rates.py first to use observed lapse rates.")

# ── load DDF scale array ──────────────────────────────────────────────────────
ddf_scale_path = Path(f"netcdf/{site}/{site}_ddf_scale.nc")
if ddf_scale_path.exists():
    ds_ddf = xr.open_dataset(ddf_scale_path)
    ddf_scale_arr = ds_ddf["ddf_scale"].values   # (365, ny, nx) float32
    ds_ddf.close()
    print(f"Loaded DDF scale array: {ddf_scale_arr.shape}, "
          f"melt-season range {np.nanmin(ddf_scale_arr[151:273]):.2f}–"
          f"{np.nanmax(ddf_scale_arr[151:273]):.2f}")
else:
    ddf_scale_arr = None
    print("DDF scale not found — using spatially uniform DDF. "
          "Run compute_ddf_scale.py first.")

# ── load CARRA temperature ────────────────────────────────────────────────────
carra_path = Path("data/CARRA/CARRA_t2m_nearest_PROMICE.nc")
print("Loading CARRA temperature...")
ds_carra = xr.open_dataset(carra_path, decode_timedelta=False)

# Select station for this site
station_names = ds_carra["name"].values.tolist()
if site not in station_names:
    sys.exit(f"Site '{site}' not found in CARRA stations: {station_names}")
st_idx = station_names.index(site)

t2m_raw    = ds_carra["t2m"].isel(station=st_idx)          # (time,), Kelvin
elev_carra = float(ds_carra["altitude_mod"].isel(station=st_idx).values)  # m

# Convert K → °C and resample 3-hourly → daily mean
t_celsius  = (t2m_raw - 273.15).rename("t2m_C")
t_daily    = t_celsius.resample(time="1D").mean()

print(f"  Station elevation (CARRA model orography): {elev_carra:.1f} m")
print(f"  Temperature range: {float(t_daily.min()):.1f} to {float(t_daily.max()):.1f} °C")
print(f"  Daily records: {len(t_daily)}")

# ── load terrain and build elevation grid ────────────────────────────────────
print("\nLoading terrain...")
terrain_path = Path(f"netcdf/{site}/{site}_terrain.nc")
terrain = xr.open_dataset(terrain_path)

# Use one masked SCF file as the reprojection template
tmpl_file = sorted(masked_dir.glob(f"{site}_scf_*.nc"))[0]
ds_tmpl   = xr.open_dataset(tmpl_file)
da_tmpl   = (ds_tmpl["snow_cover_fraction_masked"]
             .isel(time=0).drop_vars("time")
             .rio.write_crs(target_epsg))

elev_modis = (terrain["elevation"]
              .rio.write_crs(target_epsg)
              .rio.reproject_match(da_tmpl)
              .values.astype(np.float32))   # (ny, nx)

ny, nx = elev_modis.shape
print(f"  MODIS grid: {ny} × {nx}")
print(f"  Elevation range: {np.nanmin(elev_modis):.0f} – {np.nanmax(elev_modis):.0f} m")

# Temperature offset arrays per month — precomputed for speed
# ΔT = −lapse_rate(month) × Δelev / 1000
delta_t_by_month = {
    m: -lr * (elev_modis - elev_carra) / 1000.0
    for m, lr in lapse_by_month.items()
}   # dict of (ny, nx) float32 arrays, one per calendar month

# ── build glacier mask from filled dataset ────────────────────────────────────
# Take the max fill_flag across ALL time steps — any pixel ever filled is glacier.
# Using isel(time=0) fails because polar-night days have fill_flag=0 everywhere.
print("\nBuilding glacier mask from fill_flag (max over all time steps)...")
glacier_mask = np.zeros((ny, nx), dtype=bool)
for f in sorted(filled_dir.glob(f"{site}_scf_*.nc")):
    ds_tmp = xr.open_dataset(f)
    glacier_mask |= (ds_tmp["fill_flag"].max(dim="time").values == 1)
    ds_tmp.close()
print(f"  Glacier pixels: {glacier_mask.sum():,}")

# ── process year by year ──────────────────────────────────────────────────────
masked_files = sorted(masked_dir.glob(f"{site}_scf_*.nc"))
filled_files = sorted(filled_dir.glob(f"{site}_scf_*.nc"))

# Index filled files by year for quick lookup
filled_by_year = {int(f.stem.split("_")[-2]): f for f in filled_files}

annual_records = []

print("\nProcessing years...")
for nc_masked in masked_files:
    year = int(nc_masked.stem.split("_")[-1])

    if year not in filled_by_year:
        print(f"  {year}: no filled file found, skipping")
        continue

    # Load SCF data for this year
    ds_m = xr.open_dataset(nc_masked)
    ds_f = xr.open_dataset(filled_by_year[year])

    scf_masked = ds_m["snow_cover_fraction_masked"].where(
        ds_m["snow_cover_fraction_masked"] <= 100).values.astype(np.float32)  # (ndays, ny, nx)
    scf_filled = ds_f["scf_filled"].values.astype(np.float32)

    times = pd.DatetimeIndex(ds_m.time.values)
    ndays = len(times)

    # Select CARRA daily temperature for this year and build a date→value dict
    # for robust lookup (avoids datetime precision/timezone mismatches)
    t_yr = t_daily.sel(time=t_daily.time.dt.year == year)
    t_yr_dict = {pd.Timestamp(t).date(): float(v)
                 for t, v in zip(t_yr.time.values, t_yr.values)}

    # Output arrays
    snow_melt_yr = np.full((ndays, ny, nx), np.nan, dtype=np.float32)
    ice_melt_yr  = np.full((ndays, ny, nx), np.nan, dtype=np.float32)
    surf_type_yr = np.full((ndays, ny, nx), FLAG_NODATA, dtype=np.int8)

    for i, date in enumerate(times):
        # Station temperature for this day
        date_key = pd.Timestamp(date).date()
        if date_key not in t_yr_dict:
            continue   # no CARRA data for this day
        t_station = t_yr_dict[date_key]

        # Gridded temperature via month-specific lapse rate: (ny, nx)
        t_grid = t_station + delta_t_by_month[date.month]
        pdd    = np.maximum(t_grid, 0.0)      # PDD per pixel

        scf_m_day = scf_masked[i]             # land SCF (NaN on glacier/ocean)
        scf_f_day = scf_filled[i]             # filled SCF (land + glacier, NaN on ocean/polar night)

        # Initialise surface type for pixels with valid elevation (inside AOI)
        aoi = np.isfinite(elev_modis)

        # Spatially varying DDF scale for this DOY (1-indexed)
        if ddf_scale_arr is not None:
            ddf_s = ddf_scale_arr[date.dayofyear - 1]   # (ny, nx)
            ddf_s = np.where(np.isfinite(ddf_s), ddf_s, 1.0)
        else:
            ddf_s = 1.0

        # --- Land (non-glacier) pixels ---
        land_snow  = aoi & ~glacier_mask & (scf_m_day >= SNOW_THRESHOLD)
        land_bare  = aoi & ~glacier_mask & (scf_m_day < SNOW_THRESHOLD) & np.isfinite(scf_m_day)
        land_cloud = aoi & ~glacier_mask & ~np.isfinite(scf_m_day)

        snow_melt_yr[i][land_snow] = (DDF_SNOW * ddf_s * pdd)[land_snow]
        snow_melt_yr[i][land_bare] = 0.0
        surf_type_yr[i][land_snow] = FLAG_SNOW
        surf_type_yr[i][land_bare] = FLAG_BARE_LAND
        surf_type_yr[i][land_cloud] = FLAG_CLOUD

        # --- Glacier pixels ---
        glac_snow = glacier_mask & (scf_f_day >= SNOW_THRESHOLD)
        glac_ice  = glacier_mask & (scf_f_day < SNOW_THRESHOLD) & np.isfinite(scf_f_day)
        glac_nan  = glacier_mask & ~np.isfinite(scf_f_day)

        snow_melt_yr[i][glac_snow] = (DDF_SNOW * ddf_s * pdd)[glac_snow]
        ice_melt_yr[i][glac_ice]   = (DDF_ICE  * ddf_s * pdd)[glac_ice]
        surf_type_yr[i][glac_snow] = FLAG_SNOW
        surf_type_yr[i][glac_ice]  = FLAG_ICE
        surf_type_yr[i][glac_nan]  = FLAG_NODATA

    total_melt_yr = np.where(
        np.isfinite(snow_melt_yr) | np.isfinite(ice_melt_yr),
        np.nansum([snow_melt_yr, ice_melt_yr], axis=0),
        np.nan,
    ).astype(np.float32)

    # Annual totals — convert sum of mm/day over pixels to km³ w.e.
    # sum [mm] × pixel_area [m²] × 1e-3 [m/mm] × 1e-9 [km³/m³]
    pixel_area_m2 = 500.0 * 500.0
    to_km3 = pixel_area_m2 * 1e-3 * 1e-9

    snow_km3  = float(np.nansum(snow_melt_yr)) * to_km3
    ice_km3   = float(np.nansum(ice_melt_yr))  * to_km3
    total_km3 = snow_km3 + ice_km3
    annual_records.append({
        "year": year,
        "snow_melt_km3": snow_km3,
        "ice_melt_km3":  ice_km3,
        "total_melt_km3": total_km3,
    })
    print(f"  {year}: snow={snow_km3:.3f} km³ w.e.  "
          f"ice={ice_km3:.3f} km³ w.e.  "
          f"total={total_km3:.3f} km³ w.e.")

    # Write NetCDF
    ds_out = xr.Dataset(
        {
            "snow_melt":    (["time", "y", "x"], snow_melt_yr),
            "ice_melt":     (["time", "y", "x"], ice_melt_yr),
            "total_melt":   (["time", "y", "x"], total_melt_yr),
            "surface_type": (["time", "y", "x"], surf_type_yr),
        },
        coords={"time": ds_m.time, "y": ds_m.y, "x": ds_m.x},
    )
    ds_out["snow_melt"].attrs    = {"units": "mm w.e. day-1",
                                    "long_name": "Snow melt rate"}
    ds_out["ice_melt"].attrs     = {"units": "mm w.e. day-1",
                                    "long_name": "Ice melt rate"}
    ds_out["total_melt"].attrs   = {"units": "mm w.e. day-1",
                                    "long_name": "Total melt rate (snow + ice)"}
    ds_out["surface_type"].attrs = {
        "long_name": "Surface classification",
        "flag_values": "0, 1, 2, 3, -1",
        "flag_meanings": "no_data snow bare_ice bare_land cloud_or_missing",
    }
    ds_out.attrs = {
        "site":           site,
        "model":          "Degree-day",
        "lapse_rate":     "observed monthly medians (°C/km) from compute_lapse_rates.py"
                          if lapse_csv.exists() and args.lapse_rate is None
                          else f"{args.lapse_rate or LAPSE_RATE_DEFAULT} degC/km (constant)",
        "ddf_snow":       f"{DDF_SNOW} mm w.e. degC-1 day-1",
        "ddf_ice":        f"{DDF_ICE} mm w.e. degC-1 day-1",
        "snow_threshold": f"{SNOW_THRESHOLD}% NDSI",
        "carra_station":  site,
        "carra_elev_m":   elev_carra,
        "source_scf":     f"MODIS MOD10A1F masked + RF gap-filled",
        "source_temp":    "CARRA 2m temperature (3-hourly → daily mean)",
    }

    out_path = out_dir / f"{site}_melt_{year}.nc"
    ds_out.to_netcdf(out_path)
    ds_m.close(); ds_f.close()

# ── save annual totals CSV ────────────────────────────────────────────────────
df_ann = pd.DataFrame(annual_records)
csv_path = csv_dir / "melt_annual_totals.csv"
df_ann.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ── annual totals figure ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(df_ann["year"], df_ann["snow_melt_km3"],
       label="Snow melt", color="steelblue", alpha=0.8)
ax.bar(df_ann["year"], df_ann["ice_melt_km3"],
       bottom=df_ann["snow_melt_km3"],
       label="Ice melt", color="firebrick", alpha=0.8)
ax.set_xlabel("Year")
ax.set_ylabel("Total melt (km³ w.e.)")
ax.set_title(f"{site.capitalize()}: Annual snow and ice melt\n"
             f"(DDF_snow={DDF_SNOW}, DDF_ice={DDF_ICE} mm w.e. °C⁻¹ d⁻¹")
ax.legend(fontsize=9)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(fig_dir / "melt_annual_totals.png", dpi=150)
plt.close(fig)
print(f"Saved: figures/{site}/melt_annual_totals.png")

print("\nDone.")
