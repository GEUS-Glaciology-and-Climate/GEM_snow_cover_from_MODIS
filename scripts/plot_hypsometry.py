"""
Hypsometry of GEM sites with climate mast elevation and ice fraction
=====================================================================
Plots the elevation–area distribution (hypsometry) for each of the three
GEM sites (Zackenberg, Nuuk, Disko) with bars split into land (ice-free)
and glacier ice, and marks the elevation of the main ClimateBasis mast.

Data sources:
  - Elevation: ArcticDEM Mosaic v4.1 (32 m), clipped per site
      netcdf/{site}/{site}_dem_32m.tif
  - Ice mask: SDFI G250 glacier polygons
      /home/shl/mdrev/data/sdfi/G250_VEKTOR/Is.shp
  - Mast positions (lat/lon): data/station_metadata_main_masts.csv
    Mast elevation is sampled from the DEM at that position.

Output:
  figures/hypsometry_all_sites.{png,pdf}

Usage:
    python scripts/plot_hypsometry.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import rioxarray as rxr
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from pyproj import Transformer
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ICE_SHP  = "/home/shl/mdrev/data/sdfi/G250_VEKTOR/Is.shp"
MAST_CSV = "data/station_metadata_main_masts.csv"
ELEV_BIN = 50   # m
OUT_DIR  = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SITES = [
    dict(name="zackenberg", label="Zackenberg", epsg=32627,
         dem="netcdf/zackenberg/zackenberg_dem_32m.tif",
         aoi="shp/zackenberg_aoi.gpkg"),
    dict(name="disko",       label="Disko",       epsg=32622,
         dem="netcdf/disko/disko_dem_32m.tif",
         aoi="shp/disko_aoi_v2.shp"),
    dict(name="nuuk",        label="Kobbefjord", epsg=32622,
         dem="netcdf/nuuk/nuuk_dem_32m.tif",
         aoi="shp/nuuk_aoi.shp"),
]

# ---------------------------------------------------------------------------
# Pre-compute hypsometry, ice fraction, and mast elevation per site
# ---------------------------------------------------------------------------

masts = pd.read_csv(MAST_CSV)
masts["station_lower"] = masts["station"].str.lower()


def site_hypsometry(site: dict) -> dict:
    # --- DEM ---
    with rasterio.open(site["dem"]) as src:
        dem_arr = src.read(1).astype(float)
        nodata  = src.nodata
        transform = src.transform
        shape   = dem_arr.shape
    if nodata is not None:
        dem_arr[dem_arr == nodata] = np.nan

    valid = np.isfinite(dem_arr) & (dem_arr >= 0)
    elev_vals = dem_arr[valid]

    # --- Ice mask (rasterised) ---
    aoi = gpd.read_file(site["aoi"]).to_crs(epsg=site["epsg"])
    ice = gpd.read_file(ICE_SHP).to_crs(epsg=site["epsg"])
    ice["geometry"] = ice.geometry.buffer(0)   # repair invalid geometries
    ice_clip = gpd.clip(ice, aoi)

    if len(ice_clip) > 0:
        ice_raster = geometry_mask(
            ice_clip.geometry, transform=transform,
            invert=True, out_shape=shape,
        )
        is_ice = ice_raster & valid
    else:
        is_ice = np.zeros(shape, dtype=bool)

    # --- Elevation histogram split by land / ice ---
    elev_max = float(np.ceil(elev_vals.max() / 100) * 100)
    bins = np.arange(0, elev_max + ELEV_BIN, ELEV_BIN)

    counts_total, edges = np.histogram(elev_vals, bins=bins)
    counts_ice, _       = np.histogram(dem_arr[is_ice], bins=bins)
    counts_land         = counts_total - counts_ice

    total = counts_total.sum()
    land_pct = counts_land / total * 100
    ice_pct  = counts_ice  / total * 100
    bin_centres = 0.5 * (edges[:-1] + edges[1:])

    # --- Mast elevation (DEM pixel nearest to station lat/lon) ---
    dem_xr = rxr.open_rasterio(site["dem"], masked=True).squeeze()
    row = masts[masts["station_lower"] == site["name"]].iloc[0]
    tr  = Transformer.from_crs("EPSG:4326", f"EPSG:{site['epsg']}", always_xy=True)
    mx, my = tr.transform(float(row["lon"]), float(row["lat"]))
    mast_elev = float(dem_xr.sel(x=mx, y=my, method="nearest").values)

    ice_total_pct = counts_ice.sum() / total * 100

    return dict(
        bin_centres=bin_centres,
        land_pct=land_pct,
        ice_pct=ice_pct,
        elev_max=elev_max,
        mast_elev=mast_elev,
        ice_total_pct=ice_total_pct,
    )


print("Loading DEMs and computing hypsometry...")
data = [site_hypsometry(s) for s in SITES]
for s, d in zip(SITES, data):
    print(f"  {s['label']:12s}: max elev = {d['elev_max']:.0f} m, "
          f"mast = {d['mast_elev']:.0f} m a.s.l., "
          f"ice = {d['ice_total_pct']:.1f}% of AOI")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)

COLOR_LAND = "steelblue"
COLOR_ICE  = "#a8d5e2"   # pale blue for glacier ice

for ax, site, d, label in zip(axes, SITES, data, ["(a)", "(b)", "(c)"]):
    total_pct    = d["land_pct"] + d["ice_pct"]
    bar_h        = ELEV_BIN * 0.85

    # Land portion
    ax.barh(d["bin_centres"], d["land_pct"], height=bar_h,
            color=COLOR_LAND, alpha=0.80, linewidth=0)
    # Ice portion stacked on top
    ax.barh(d["bin_centres"], d["ice_pct"], height=bar_h,
            left=d["land_pct"], color=COLOR_ICE, alpha=0.90,
            linewidth=0.3, edgecolor="white")

    # Mast elevation line + label
    ax.axhline(d["mast_elev"], color="firebrick", linewidth=1.5,
               linestyle="--", zorder=5)
    ax.text(total_pct.max() * 0.97, d["mast_elev"] + d["elev_max"] * 0.02,
            f"mast: {d['mast_elev']:.0f} m",
            va="bottom", ha="right", fontsize=8, color="firebrick")

    ax.set_title(site["label"], fontsize=11)
    ax.text(0.03, 0.97, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")
    ax.set_xlabel("Area (%)", fontsize=9)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)

shared_elev_max = max(d["elev_max"] for d in data)
shared_pct_max  = max((d["land_pct"] + d["ice_pct"]).max() for d in data)
axes[0].set_ylim(0, shared_elev_max)
axes[0].set_xlim(0, 12)
axes[0].set_ylabel("Elevation (m a.s.l.)", fontsize=9)

# Shared legend
land_patch = mpatches.Patch(color=COLOR_LAND, alpha=0.80, label="Ice-free land")
ice_patch  = mpatches.Patch(color=COLOR_ICE,  alpha=0.90, label="Glacier ice")
axes[1].legend(handles=[land_patch, ice_patch], fontsize=8,
               loc="upper right", framealpha=0.85)

fig.tight_layout()

for ext in ("png", "pdf"):
    out = OUT_DIR / f"hypsometry_all_sites.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close(fig)
