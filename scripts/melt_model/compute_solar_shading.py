"""
Terrain Shadow Geometry for AWS Station Points
===============================================
For each AWS station (ZAC_L, ZAC_U), computes:
  1. Horizon elevation angle profile (0–360° azimuth) using the 32 m ArcticDEM
  2. Hourly solar position (pvlib) at each station for each day of year
  3. Sunlit fraction per day of year (fraction of daylight hours not in shadow)
  4. Mean potential clear-sky direct radiation per day of year

Outputs
-------
  results/csvs/{site}/solar_shading_stations.csv   — daily DOY × station table
  figures/{site}/solar_shading_horizon.png          — horizon profiles
  figures/{site}/solar_shading_sunlit_fraction.png  — sunlit fraction by DOY

Usage
-----
  python scripts/melt_model/compute_solar_shading.py --site zackenberg
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import rioxarray
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pvlib
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

# ── args & config ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True)
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site        = cfg["site"]
target_epsg = cfg["target_epsg"]
fig_dir     = Path(f"figures/{site}/")
csv_dir     = Path(f"results/csvs/{site}/")
fig_dir.mkdir(parents=True, exist_ok=True)
csv_dir.mkdir(parents=True, exist_ok=True)

# ── station definitions ───────────────────────────────────────────────────────
pos = pd.read_csv("data/positions.csv")
pos.columns = pos.columns.str.strip()
pos.index   = pos.iloc[:, 0].str.strip()
pos = pos.iloc[:, 1:]

stations = {
    "ZAC_L": {"color": "steelblue"},
    "ZAC_U": {"color": "firebrick"},
}
for name in stations:
    stations[name]["lat"]  = float(pos.loc[name, "Latitude"])
    stations[name]["lon"]  = float(pos.loc[name, "Longitude"])
    stations[name]["elev"] = float(pos.loc[name, "Elevation"])

# ── load 32 m DEM ─────────────────────────────────────────────────────────────
dem_path = Path(f"netcdf/{site}/{site}_dem_32m.tif")
print(f"Loading 32 m DEM: {dem_path}")
dem_da = rioxarray.open_rasterio(dem_path, masked=True).squeeze()
dem_x  = dem_da.x.values.astype(np.float64)
dem_y  = dem_da.y.values.astype(np.float64)
dem_z  = dem_da.values.astype(np.float64)

# Fill NaN (ocean) with 0 — ocean horizon is always at or below sea level
dem_z  = np.where(np.isfinite(dem_z), dem_z, 0.0)
res    = abs(float(dem_da.rio.resolution()[0]))   # 32 m

print(f"  DEM size: {dem_z.shape}, resolution: {res:.0f} m")
print(f"  Elevation range: {dem_z.min():.0f} – {dem_z.max():.0f} m")

# Build interpolator — note y is decreasing (north-up), so flip for interp
# RegularGridInterpolator requires strictly increasing axes
dem_interp = RegularGridInterpolator(
    (dem_y[::-1], dem_x),
    dem_z[::-1, :],
    method="linear",
    bounds_error=False,
    fill_value=0.0,
)

# Convert station lat/lon to UTM for DEM lookup
from pyproj import Transformer
to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)

for name, info in stations.items():
    x_utm, y_utm = to_utm.transform(info["lon"], info["lat"])
    info["x_utm"] = x_utm
    info["y_utm"] = y_utm
    print(f"  {name}: UTM ({x_utm:.0f}, {y_utm:.0f}), elev={info['elev']:.0f} m")

# ── compute horizon profile ───────────────────────────────────────────────────
N_AZ       = 360          # azimuth directions (1° steps)
MAX_DIST   = 20_000       # m — scan 20 km outward
STEP       = res * 2      # scan step = 2 × DEM pixel = 64 m

print("\nComputing horizon profiles...")

def compute_horizon(x0, y0, z0, n_az=N_AZ, max_dist=MAX_DIST, step=STEP):
    """Return horizon elevation angle (degrees) for each azimuth 0..n_az-1."""
    azimuths = np.linspace(0, 360, n_az, endpoint=False)
    horizons = np.zeros(n_az)
    for i, az in enumerate(azimuths):
        az_rad = np.radians(az)
        dx = np.sin(az_rad)   # east component
        dy = np.cos(az_rad)   # north component
        distances = np.arange(step, max_dist + step, step)
        xs = x0 + dx * distances
        ys = y0 + dy * distances
        pts = np.column_stack([ys, xs])
        z_far = dem_interp(pts)
        elev_angles = np.degrees(np.arctan2(z_far - z0, distances))
        horizons[i] = max(0.0, elev_angles.max())
    return azimuths, horizons

horizon_profiles = {}
for name, info in stations.items():
    azimuths, horizons = compute_horizon(info["x_utm"], info["y_utm"], info["elev"])
    horizon_profiles[name] = {"azimuths": azimuths, "horizons": horizons}
    print(f"  {name}: mean horizon angle = {horizons.mean():.2f}°, "
          f"max = {horizons.max():.1f}° at {azimuths[horizons.argmax()]:.0f}°")

# ── solar position and shading by DOY ────────────────────────────────────────
print("\nComputing solar position and shading fraction by DOY...")

# Use a representative year (non-leap)
YEAR = 2015
records = []

for name, info in stations.items():
    location = pvlib.location.Location(
        latitude=info["lat"], longitude=info["lon"],
        altitude=info["elev"], tz="UTC"
    )
    hz = horizon_profiles[name]["horizons"]   # (360,)

    for doy in range(1, 366):
        date = pd.Timestamp(f"{YEAR}-01-01") + pd.Timedelta(days=doy - 1)
        # Hourly timestamps for this day
        times = pd.date_range(date, periods=24, freq="1h", tz="UTC")
        solpos = location.get_solarposition(times)

        # Daylight hours: sun above horizon
        daylight = solpos["elevation"] > 0
        n_daylight = daylight.sum()
        if n_daylight == 0:
            records.append({"station": name, "doy": doy,
                            "sunlit_frac": 0.0, "n_daylight": 0,
                            "n_sunlit": 0, "mean_solar_elev": 0.0})
            continue

        # For each daylight hour, check if sun is above terrain horizon
        az = solpos.loc[daylight, "azimuth"].values       # 0–360
        el = solpos.loc[daylight, "elevation"].values      # degrees

        # Interpolate horizon angle at solar azimuth
        az_idx = (az % 360).astype(int)
        horizon_at_sun = hz[az_idx]

        sunlit = el > horizon_at_sun
        n_sunlit = sunlit.sum()

        records.append({
            "station":        name,
            "doy":            doy,
            "sunlit_frac":    n_sunlit / n_daylight,
            "n_daylight":     int(n_daylight),
            "n_sunlit":       int(n_sunlit),
            "mean_solar_elev": float(el.mean()),
        })

df = pd.DataFrame(records)
csv_path = csv_dir / "solar_shading_stations.csv"
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# ── print melt season summary ─────────────────────────────────────────────────
print("\nMelt season (DOY 152–273, Jun–Sep) mean sunlit fraction:")
melt = df[df["doy"].between(152, 273)]
for name in stations:
    sf = melt[melt["station"] == name]["sunlit_frac"].mean()
    print(f"  {name}: {sf:.3f} ({sf*100:.1f}% of daylight hours sunlit)")

print("\nImplied DDF scaling (relative to ZAC_U):")
sf_u = melt[melt["station"] == "ZAC_U"]["sunlit_frac"].mean()
for name in stations:
    sf = melt[melt["station"] == name]["sunlit_frac"].mean()
    print(f"  {name}: DDF scale factor = {sf/sf_u:.3f}")

# ── Figure 1: Horizon profiles ────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5), subplot_kw={"projection": "polar"})
ax1 = plt.subplot(111, projection="polar")

for name, info in stations.items():
    az_rad = np.radians(horizon_profiles[name]["azimuths"])
    hz     = horizon_profiles[name]["horizons"]
    # Polar plot: 0=N, clockwise → need to convert
    ax1.plot(np.pi/2 - az_rad, hz, linewidth=1.5,
             color=info["color"], label=name)
    ax1.fill_between(np.pi/2 - az_rad, 0, hz,
                     color=info["color"], alpha=0.15)

ax1.set_theta_zero_location("N")
ax1.set_theta_direction(-1)
ax1.set_xlabel("Horizon elevation angle (°)", labelpad=15)
ax1.set_title(f"{site.capitalize()}: Terrain horizon profiles\n(0° = north, clockwise)",
              fontsize=10, pad=20)
ax1.legend(loc="lower right", fontsize=9, bbox_to_anchor=(1.3, -0.1))
ax1.grid(True, alpha=0.4)
fig1.tight_layout()
fig1.savefig(fig_dir / "solar_shading_horizon.png", dpi=150)
plt.close(fig1)
print(f"Saved: figures/{site}/solar_shading_horizon.png")

# ── Figure 2: Sunlit fraction by DOY ─────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax_sf, ax_el = axes2

for name, info in stations.items():
    sub = df[df["station"] == name]
    ax_sf.plot(sub["doy"], sub["sunlit_frac"] * 100,
               color=info["color"], linewidth=1.5, label=name)
    ax_el.plot(sub["doy"], sub["mean_solar_elev"],
               color=info["color"], linewidth=1.5, label=name)

# Shade melt season
for ax in axes2:
    ax.axvspan(152, 273, alpha=0.08, color="gold", label="Melt season (Jun–Sep)")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)

ax_sf.set_ylabel("Sunlit fraction (% of daylight hours)", fontsize=9)
ax_sf.set_title(f"{site.capitalize()}: Solar shading by day of year", fontsize=10)
ax_el.set_ylabel("Mean solar elevation during daylight (°)", fontsize=9)
ax_el.set_xlabel("Day of year", fontsize=9)
ax_el.set_xlim(1, 365)
ax_el.xaxis.set_major_locator(ticker.MultipleLocator(30))

fig2.tight_layout()
fig2.savefig(fig_dir / "solar_shading_sunlit_fraction.png", dpi=150)
plt.close(fig2)
print(f"Saved: figures/{site}/solar_shading_sunlit_fraction.png")

print("\nDone.")
