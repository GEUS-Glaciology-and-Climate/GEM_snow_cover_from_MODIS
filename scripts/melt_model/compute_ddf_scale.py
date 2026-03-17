"""
Spatially and Temporally Varying DDF Scale Factor
==================================================
Pre-computes a DDF scaling array (ny, nx, 365) based on the ratio of
potential clear-sky direct radiation on each MODIS pixel surface to that
on a flat horizontal surface at the same location.

Physics
-------
The degree-day factor is proportional to absorbed shortwave radiation.
For a tilted surface at slope β and aspect α, the direct radiation
intensity relative to a horizontal surface scales as:

    cos(θ) = sin(el)·cos(β) + cos(el)·sin(β)·cos(az_sun − α)

where el = solar elevation, az_sun = solar azimuth.

The daily scale factor for pixel (y,x) on day doy is:

    scale(y, x, doy) = Σ_h max(0, cos(θ)) / Σ_h max(0, sin(el))

where the sum is over hourly time steps with el > 0 (daylight).
Pixels facing away from the sun (cos(θ) < 0) contribute zero.

Terrain shading is optionally applied using pre-computed horizon angles
at each MODIS pixel (computed at MODIS 500 m resolution from the 32 m DEM).

The output scale is normalised to the catchment mean over the melt season
so that the reference DDF values (DDF_snow, DDF_ice) represent the
catchment-average, not an arbitrary flat surface.

Outputs
-------
  netcdf/{site}/{site}_ddf_scale.nc
    ddf_scale (doy, y, x)  — dimensionless, float32
    Attributes include mean ± std over melt season

  figures/{site}/ddf_scale_mean_melt_season.png
  figures/{site}/ddf_scale_timeseries_stations.png

Usage
-----
  python scripts/melt_model/compute_ddf_scale.py --site zackenberg
  python scripts/melt_model/compute_ddf_scale.py --site zackenberg --no-shading
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import pvlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from pyproj import Transformer

# ── args & config ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True)
parser.add_argument("--no-shading", action="store_true",
                    help="Skip terrain shading (slope/aspect only)")
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site        = cfg["site"]
target_epsg = cfg["target_epsg"]
out_dir     = Path(f"netcdf/{site}/")
fig_dir     = Path(f"figures/{site}/")
fig_dir.mkdir(parents=True, exist_ok=True)

USE_SHADING = not args.no_shading
MELT_DOY_START, MELT_DOY_END = 152, 273   # Jun 1 – Sep 30
N_AZ    = 72     # horizon azimuth resolution (5° steps) — coarser than station script
MAX_DIST = 15_000  # m
STEP_M  = 500    # match MODIS pixel size for horizon scan

# ── load terrain ──────────────────────────────────────────────────────────────
terrain_path = Path(f"netcdf/{site}/{site}_terrain.nc")
print("Loading terrain and resampling to MODIS 500 m grid...")
terrain = xr.open_dataset(terrain_path)

# Resample 32m terrain to MODIS 500m grid using a masked SCF file as template
masked_dir = Path(f"netcdf/{site}_masked/")
tmpl_file  = sorted(masked_dir.glob(f"{site}_scf_*.nc"))[0]
ds_tmpl    = xr.open_dataset(tmpl_file)
da_tmpl    = (ds_tmpl["snow_cover_fraction_masked"]
              .isel(time=0).drop_vars("time")
              .rio.write_crs(target_epsg))

elev_da   = terrain["elevation"].rio.write_crs(target_epsg).rio.reproject_match(da_tmpl)
slope_da  = terrain["slope"].rio.write_crs(target_epsg).rio.reproject_match(da_tmpl)
aspect_da = terrain["aspect"].rio.write_crs(target_epsg).rio.reproject_match(da_tmpl)

elev_2d   = elev_da.values.astype(np.float32)
slope_2d  = np.radians(slope_da.values.astype(np.float32))
aspect_2d = np.radians(aspect_da.values.astype(np.float32))

x_coords = da_tmpl.x.values.astype(np.float64)
y_coords = da_tmpl.y.values.astype(np.float64)
ny, nx   = elev_2d.shape
res      = abs(float(x_coords[1] - x_coords[0]))
ds_tmpl.close()

print(f"  Grid: {ny} × {nx}, resolution: {res:.0f} m")

# Representative lat/lon for solar position (use grid centre — variation
# across a ~150×130 km domain at 74°N is < 0.1° in solar angles)
to_wgs84 = Transformer.from_crs(f"EPSG:{target_epsg}", "EPSG:4326", always_xy=True)
cx = float(x_coords[nx // 2])
cy = float(y_coords[ny // 2])
centre_lon, centre_lat = to_wgs84.transform(cx, cy)
print(f"  Centre: {centre_lat:.3f}°N, {centre_lon:.3f}°E")

# Pre-flatten terrain arrays as float32 to keep memory low
slope_f  = slope_2d.ravel().astype(np.float32)
aspect_f = aspect_2d.ravel().astype(np.float32)
valid    = np.isfinite(elev_2d.ravel())

# ── terrain shading: horizon angles at each MODIS pixel ──────────────────────
if USE_SHADING:
    print("\nComputing terrain horizon profiles at MODIS pixels "
          f"({N_AZ} azimuths, {MAX_DIST/1000:.0f} km range)...")

    # Build a 2D DEM interpolator from the MODIS grid for the horizon scan
    # (use the same 500m grid — coarse but consistent)
    dem_interp = RegularGridInterpolator(
        (y_coords[::-1], x_coords),
        elev_2d[::-1, :],
        method="linear", bounds_error=False, fill_value=0.0,
    )

    az_steps = np.linspace(0, 2 * np.pi, N_AZ, endpoint=False)  # radians
    distances = np.arange(STEP_M, MAX_DIST + STEP_M, STEP_M)

    # horizon_grid: (N_AZ, ny, nx) — elevation angle in radians for each azimuth
    horizon_grid = np.zeros((N_AZ, ny, nx), dtype=np.float32)

    xx, yy = np.meshgrid(x_coords, y_coords)  # (ny, nx)
    z_flat = elev_2d.ravel()

    for i, az in enumerate(az_steps):
        dx = np.sin(az)
        dy = np.cos(az)
        max_angles = np.zeros(ny * nx, dtype=np.float64)

        for d in distances:
            xs_far = xx.ravel() + dx * d
            ys_far = yy.ravel() + dy * d
            pts = np.column_stack([ys_far, xs_far])
            z_far = dem_interp(pts)
            angles = np.arctan2(z_far - z_flat, d)
            max_angles = np.maximum(max_angles, angles)

        horizon_grid[i] = max_angles.reshape(ny, nx)

        if (i + 1) % 12 == 0:
            print(f"  {i+1}/{N_AZ} azimuths done")

    az_steps_deg = np.degrees(az_steps)
    print("  Horizon profiles complete.")
else:
    print("\nSkipping terrain shading (--no-shading)")
    horizon_grid = None

# ── compute DDF scale: one DOY at a time, write incrementally ─────────────────
print("\nComputing DDF scale (one DOY at a time)...")

YEAR = 2015
location = pvlib.location.Location(
    latitude=centre_lat, longitude=centre_lon,
    altitude=float(np.nanmean(elev_2d[np.isfinite(elev_2d)])), tz="UTC"
)

# Pre-allocate output — but compute and accumulate melt-season mean separately
# to avoid holding all 365 slices simultaneously
out_path = out_dir / f"{site}_ddf_scale.nc"

ddf_slices = []   # list of (ny, nx) float32 arrays, one per DOY

for doy in range(1, 366):
    date  = pd.Timestamp(f"{YEAR}-01-01") + pd.Timedelta(days=doy - 1)
    times = pd.date_range(date, periods=24, freq="1h", tz="UTC")
    sp    = location.get_solarposition(times)

    el_rad   = np.radians(sp["elevation"].values).astype(np.float32)
    az_rad   = np.radians(sp["azimuth"].values).astype(np.float32)
    daylight = sp["elevation"].values > 0

    if daylight.sum() == 0:
        ddf_slices.append(np.full((ny, nx), np.nan, dtype=np.float32))
        continue

    ref_sum = float(np.sum(np.maximum(0.0, np.sin(el_rad[daylight]))))
    pix_sum = np.zeros(ny * nx, dtype=np.float32)

    for h in np.where(daylight)[0]:
        el = float(el_rad[h])
        az = float(az_rad[h])
        cos_inc = (np.sin(el) * np.cos(slope_f)
                   + np.cos(el) * np.sin(slope_f) * np.cos(az - aspect_f))

        if USE_SHADING and horizon_grid is not None:
            az_deg  = np.degrees(az) % 360
            bin_idx = int(round(az_deg / (360 / N_AZ))) % N_AZ
            shaded  = el < horizon_grid[bin_idx].ravel()
            cos_inc = np.where(shaded, 0.0, cos_inc)

        pix_sum += np.maximum(0.0, cos_inc)

    with np.errstate(invalid="ignore", divide="ignore"):
        scale_2d = np.where(
            valid.reshape(ny, nx) & (ref_sum > 0),
            pix_sum.reshape(ny, nx) / ref_sum,
            np.nan,
        ).astype(np.float32)

    ddf_slices.append(scale_2d)

    if doy % 30 == 0:
        print(f"  DOY {doy}/365 — mean scale: {np.nanmean(scale_2d):.3f}")

ddf_scale = np.stack(ddf_slices, axis=0)   # (365, ny, nx), float32

# Drop the incremental melt_sum — recompute melt_mean directly from the array
# to avoid NaN/zero inconsistency in the incremental accumulation
del ddf_slices

# ── normalise to catchment mean over melt season ──────────────────────────────
melt_mean = float(np.nanmean(ddf_scale[MELT_DOY_START - 1:MELT_DOY_END]))
print(f"\nCatchment mean DDF scale (melt season): {melt_mean:.4f}")
print("Normalising to catchment mean...")
ddf_scale /= melt_mean
print(f"  After normalisation — melt season mean: "
      f"{np.nanmean(ddf_scale[MELT_DOY_START-1:MELT_DOY_END]):.3f}")
print(f"  Range (5th–95th pct): "
      f"{np.nanpercentile(ddf_scale[MELT_DOY_START-1:MELT_DOY_END], 5):.3f} – "
      f"{np.nanpercentile(ddf_scale[MELT_DOY_START-1:MELT_DOY_END], 95):.3f}")

# ── write NetCDF ──────────────────────────────────────────────────────────────
doy_coords = np.arange(1, 366)
ds_out = xr.Dataset(
    {"ddf_scale": (["doy", "y", "x"], ddf_scale)},
    coords={"doy": doy_coords, "y": da_tmpl.y, "x": da_tmpl.x},
)
ds_out["ddf_scale"].attrs = {
    "long_name": "DDF scale factor (radiation-based)",
    "units":     "dimensionless",
    "description": (
        "Ratio of potential direct radiation on pixel surface to catchment-mean "
        "over melt season. Accounts for slope, aspect" +
        (" and terrain shading." if USE_SHADING else " only (no terrain shading).")
    ),
}
ds_out.attrs["site"]             = site
ds_out.attrs["normalisation"]    = f"catchment mean over DOY {MELT_DOY_START}–{MELT_DOY_END}"
ds_out.attrs["terrain_shading"]  = str(USE_SHADING)
ds_out.attrs["melt_season_mean"] = f"{melt_mean:.4f} (pre-normalisation)"

out_path = out_dir / f"{site}_ddf_scale.nc"
ds_out.to_netcdf(out_path)
print(f"\nSaved: {out_path}")

# ── Figure 1: Mean DDF scale over melt season ─────────────────────────────────
mean_melt = np.nanmean(ddf_scale[MELT_DOY_START - 1:MELT_DOY_END], axis=0)

fig1, ax1 = plt.subplots(figsize=(8, 7))
im = ax1.imshow(mean_melt, origin="upper", cmap="RdBu_r", vmin=0.5, vmax=1.5,
                extent=[x_coords[0]/1000, x_coords[-1]/1000,
                        y_coords[-1]/1000, y_coords[0]/1000])
plt.colorbar(im, ax=ax1, label="DDF scale factor")
ax1.set_xlabel("Easting (km)")
ax1.set_ylabel("Northing (km)")
ax1.set_title(f"{site.capitalize()}: Mean DDF scale factor (Jun–Sep)\n"
              f"{'slope + aspect + shading' if USE_SHADING else 'slope + aspect only'}",
              fontsize=10)
fig1.tight_layout()
fig1.savefig(fig_dir / "ddf_scale_mean_melt_season.png", dpi=150)
plt.close(fig1)
print(f"Saved: figures/{site}/ddf_scale_mean_melt_season.png")

# ── Figure 2: DDF scale at station pixels through the year ────────────────────
pos = pd.read_csv("data/positions.csv")
pos.columns = pos.columns.str.strip()
pos.index   = pos.iloc[:, 0].str.strip()
pos = pos.iloc[:, 1:]

to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
station_colors = {"ZAC_L": "steelblue", "ZAC_U": "firebrick"}

fig2, ax2 = plt.subplots(figsize=(10, 5))
for name, color in station_colors.items():
    if name not in pos.index:
        continue
    lat = float(pos.loc[name, "Latitude"])
    lon = float(pos.loc[name, "Longitude"])
    x_utm, y_utm = to_utm.transform(lon, lat)
    col = int(np.argmin(np.abs(x_coords - x_utm)))
    row = int(np.argmin(np.abs(y_coords - y_utm)))
    scale_ts = ddf_scale[:, row, col]
    ax2.plot(doy_coords, scale_ts, color=color, linewidth=1.5, label=name)

ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax2.axvspan(MELT_DOY_START, MELT_DOY_END, alpha=0.08, color="gold", label="Melt season")
ax2.set_xlabel("Day of year", fontsize=9)
ax2.set_ylabel("DDF scale factor", fontsize=9)
ax2.set_title(f"{site.capitalize()}: DDF scale factor at station pixels by DOY\n"
              f"{'slope + aspect + shading' if USE_SHADING else 'slope + aspect only'}",
              fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 365)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(30))
fig2.tight_layout()
fig2.savefig(fig_dir / "ddf_scale_timeseries_stations.png", dpi=150)
plt.close(fig2)
print(f"Saved: figures/{site}/ddf_scale_timeseries_stations.png")

print("\nDone.")
