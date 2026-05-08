#!/usr/bin/env python3
"""
Three-panel site maps (Zackenberg UTM27N, Disko & Nuuk UTM22N).
Main panel: SDFI basemap + AOI outline + graticule.
Inset:      zoomed pixel grid derived from the masked NetCDF coordinates.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr
import matplotlib.patheffects as pe
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from shapely.geometry import LineString, MultiPoint, Point, box as shapely_box

workingdir = Path("/home/shl/mdrev/projects/modis/snow_cover")
SDFI_DIR   = Path("/home/shl/mdrev/data/sdfi/G250_VEKTOR")

SITES = {
    "Zackenberg": {
        "nc_dir":    workingdir / "netcdf" / "zackenberg_masked",
        "aoi_shp":   workingdir / "shp"    / "zackenberg_aoi.shp",
        "small_shp": workingdir / "shp"    / "zackenberg_aoi_small.shp",
        "crs":       "EPSG:32627",
        "lat_int":   0.5,
        "lon_int":   1.0,
        "inset_loc": [0.01, 0.57, 0.42, 0.42],  # [x0, y0, w, h] in axes fraction
    },
    "Disko": {
        "nc_dir":    workingdir / "netcdf" / "disko_masked",
        "aoi_shp":   workingdir / "shp"    / "disko_aoi.shp",
        "small_shp": workingdir / "shp"    / "disko_aoi_small.shp",
        "crs":       "EPSG:32622",
        "lat_int":   1.0,
        "lon_int":   2.0,
        "inset_loc": [0.01, 0.63, 0.35, 0.35],
    },
    "Nuuk": {
        "nc_dir":    workingdir / "netcdf" / "nuuk_masked",
        "aoi_shp":   workingdir / "shp"    / "nuuk_aoi.shp",
        "small_shp": workingdir / "shp"    / "nuuk_aoi_small.shp",
        "crs":       "EPSG:32622",
        "lat_int":   1.0,
        "lon_int":   1.0,
        "inset_loc": [0.57, 0.57, 0.42, 0.42],
    },
}

# ---------------------------------------------------------------------------
# Basemap — load once, repair invalid geometries
# ---------------------------------------------------------------------------
_proximity = pd.read_csv(workingdir / "data" / "mast_locations_proximity.csv")
_masts     = pd.read_csv(workingdir / "data" / "station_metadata_main_masts.csv")

print("Loading SDFI basemap…")
_land_raw    = gpd.read_file(SDFI_DIR / "Land.shp")
_land_raw["geometry"] = _land_raw.buffer(0)
_ice_raw     = gpd.read_file(SDFI_DIR / "Is.shp")
_ice_raw["geometry"]  = _ice_raw.buffer(0)
_rivers_raw  = gpd.read_file(SDFI_DIR / "Elve.shp")       # river/stream centrelines
_valleys_raw = gpd.read_file(SDFI_DIR / "FLODDAL.shp")    # river-valley polygons
_valleys_raw["geometry"] = _valleys_raw.buffer(0)
_lakes_raw   = gpd.read_file(SDFI_DIR / "Soe.shp")        # lakes and ponds
_lakes_raw["geometry"] = _lakes_raw.buffer(0)


def _clip_basemap(crs, xmin, ymin, xmax, ymax, pad=10_000):
    clip = gpd.GeoDataFrame(
        geometry=[shapely_box(xmin - pad, ymin - pad, xmax + pad, ymax + pad)],
        crs=crs,
    )
    return (
        gpd.clip(_land_raw.to_crs(crs),    clip),
        gpd.clip(_ice_raw.to_crs(crs),     clip),
        gpd.clip(_rivers_raw.to_crs(crs),  clip),
        gpd.clip(_valleys_raw.to_crs(crs), clip),
        gpd.clip(_lakes_raw.to_crs(crs),   clip),
    )


# ---------------------------------------------------------------------------
# Pixel-grid LineCollection from a subset of x/y pixel centres
# ---------------------------------------------------------------------------

def _grid_line_collection(x, y, res_x, res_y_abs, **lc_kw):
    """
    Build (nx+1) vertical + (ny+1) horizontal edge lines for a regular raster.
    x, y must be 1-D arrays of pixel centres (y decreasing).
    """
    x_edges = np.linspace(x[0]  - res_x     / 2,
                          x[-1] + res_x     / 2,
                          len(x) + 1)
    y_edges = np.linspace(y[0]  + res_y_abs / 2,   # top edge
                          y[-1] - res_y_abs / 2,   # bottom edge
                          len(y) + 1)

    xmin_g, xmax_g = x_edges[0],  x_edges[-1]
    ymin_g, ymax_g = y_edges[-1], y_edges[0]

    segs = (
        [[(xe, ymin_g), (xe, ymax_g)] for xe in x_edges]   # verticals
        + [[(xmin_g, ye), (xmax_g, ye)] for ye in y_edges]  # horizontals
    )
    return LineCollection(segs, **lc_kw)


# ---------------------------------------------------------------------------
# Graticule helpers
# ---------------------------------------------------------------------------

def _make_graticule(lats, lons, crs_to, lat_range, lon_range, n=400):
    tf = pyproj.Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)
    parallels, meridians = {}, {}
    for lat in lats:
        pts = [tf.transform(lon, lat) for lon in np.linspace(*lon_range, n)]
        pts = [(px, py) for px, py in pts if np.isfinite(px) and np.isfinite(py)]
        if len(pts) > 1:
            parallels[lat] = LineString(pts)
    for lon in lons:
        pts = [tf.transform(lon, lat) for lat in np.linspace(*lat_range, n)]
        pts = [(px, py) for px, py in pts if np.isfinite(px) and np.isfinite(py)]
        if len(pts) > 1:
            meridians[lon] = LineString(pts)
    return parallels, meridians


def _border_intersections(line, border):
    isect = line.intersection(border)
    if isect.is_empty:
        return []
    if isinstance(isect, Point):
        return [(isect.x, isect.y)]
    if isinstance(isect, MultiPoint):
        return [(p.x, p.y) for p in isect.geoms]
    if hasattr(isect, "coords"):
        return [isect.coords[0]]
    return []


def _fmt_lon(v):
    v = round(v, 4)
    s = f"{abs(v):.0f}" if abs(v) == int(abs(v)) else f"{abs(v):.1f}"
    return f"{s}°W" if v < 0 else (f"{s}°E" if v else "0°")


def _fmt_lat(v):
    v = round(v, 4)
    s = f"{v:.0f}" if v == int(v) else f"{v:.1f}"
    return f"{s}°N"


def _add_graticule_labels(ax, parallels, meridians, xmin, xmax, ymin, ymax):
    span     = min(xmax - xmin, ymax - ymin)
    tick_len = span * 0.018
    lab_off  = span * 0.013
    eps = 500
    borders = {
        "left":   LineString([(xmin, ymin - eps), (xmin, ymax + eps)]),
        "right":  LineString([(xmax, ymin - eps), (xmax, ymax + eps)]),
        "bottom": LineString([(xmin - eps, ymin), (xmax + eps, ymin)]),
        "top":    LineString([(xmin - eps, ymax), (xmax + eps, ymax)]),
    }
    fkw = dict(fontsize=6.5, clip_on=False)
    for lat, line in parallels.items():
        for side in ("left", "right"):
            for px, py in _border_intersections(line, borders[side]):
                if not (ymin <= py <= ymax):
                    continue
                if side == "left":
                    ax.plot([px, px + tick_len], [py, py],
                            color="black", lw=0.7, clip_on=False, zorder=8)
                    ax.text(px - lab_off, py, _fmt_lat(lat),
                            ha="right", va="center", **fkw)
                else:
                    ax.plot([px - tick_len, px], [py, py],
                            color="black", lw=0.7, clip_on=False, zorder=8)
                    ax.text(px + lab_off, py, _fmt_lat(lat),
                            ha="left", va="center", **fkw)
    for lon, line in meridians.items():
        for side in ("bottom", "top"):
            for px, py in _border_intersections(line, borders[side]):
                if not (xmin <= px <= xmax):
                    continue
                if side == "bottom":
                    ax.plot([px, px], [py, py + tick_len],
                            color="black", lw=0.7, clip_on=False, zorder=8)
                    ax.text(px, py - lab_off, _fmt_lon(lon),
                            ha="center", va="top", rotation=45, **fkw)
                else:
                    ax.plot([px, px], [py - tick_len, py],
                            color="black", lw=0.7, clip_on=False, zorder=8)
                    ax.text(px, py + lab_off, _fmt_lon(lon),
                            ha="center", va="bottom", rotation=45, **fkw)


# ---------------------------------------------------------------------------
# One figure per site
# ---------------------------------------------------------------------------
outdir = workingdir / "figures"
outdir.mkdir(exist_ok=True)

for name, cfg in SITES.items():
    fig, ax = plt.subplots(figsize=(7, 7))
    crs = cfg["crs"]
    print(f"Rendering {name}…")

    # --- NetCDF grid metadata -------------------------------------------
    ncfile    = sorted(cfg["nc_dir"].glob("*.nc"))[0]
    ds        = xr.open_dataset(ncfile)
    x         = ds.x.values
    y         = ds.y.values
    res_x     = float(ds.rio.resolution()[0])
    res_y_abs = abs(float(ds.rio.resolution()[1]))
    xmin_r, ymin_r, xmax_r, ymax_r = ds.rio.bounds()
    ds.close()

    # --- Main map basemap -----------------------------------------------
    land, ice, rivers, valleys, lakes = _clip_basemap(crs, xmin_r, ymin_r, xmax_r, ymax_r)
    ax.set_facecolor("#cde2f0")
    land.plot(   ax=ax, color="#e8e0ce", edgecolor="#555555", linewidth=0.3, zorder=2)
    ice.plot(    ax=ax, color="#eef2ff", edgecolor="#9999cc", linewidth=0.2, zorder=3)
    if not valleys.empty:
        valleys.plot(ax=ax, color="#b8d4e8", edgecolor="none",              zorder=4)
    if not lakes.empty:
        lakes.plot(  ax=ax, color="#cde2f0", edgecolor="#6baed6", linewidth=0.3, zorder=5)
    if not rivers.empty:
        rivers.plot( ax=ax, color="#4292c6", linewidth=0.4,                 zorder=6)

    # Main AOI boundary
    #gpd.read_file(cfg["aoi_shp"]).to_crs(crs).plot(
    #    ax=ax, facecolor="none", edgecolor="none", zorder=6,
    #)

    ax.set_xlim(xmin_r, xmax_r)
    ax.set_ylim(ymin_r, ymax_r)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.set_title(name, fontsize=10, fontweight="bold", pad=24)

    # Main graticule
    tf_fwd  = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    tf_inv  = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    corners = [tf_inv.transform(px, py) for px, py in
               [(xmin_r, ymin_r), (xmax_r, ymin_r), (xmin_r, ymax_r), (xmax_r, ymax_r)]]
    lons_c  = [c[0] for c in corners]
    lats_c  = [c[1] for c in corners]
    li, loi = cfg["lat_int"], cfg["lon_int"]
    lats_g  = np.arange(np.ceil(min(lats_c) / li)  * li,  max(lats_c) + li,  li)
    lons_g  = np.arange(np.ceil(min(lons_c) / loi) * loi, max(lons_c) + loi, loi)
    parallels, meridians = _make_graticule(
        lats_g, lons_g, crs,
        lat_range=(min(lats_c) - 0.5, max(lats_c) + 0.5),
        lon_range=(min(lons_c) - 1,   max(lons_c) + 1),
    )
    for line in parallels.values():
        xy = np.array(line.coords)
        ax.plot(xy[:, 0], xy[:, 1], "--", color="#888888", lw=0.35, alpha=0.6, zorder=1)
    for line in meridians.values():
        xy = np.array(line.coords)
        ax.plot(xy[:, 0], xy[:, 1], "--", color="#888888", lw=0.35, alpha=0.6, zorder=1)
    _add_graticule_labels(ax, parallels, meridians, xmin_r, xmax_r, ymin_r, ymax_r)

    # Main mast marker on the main map
    mast = _masts[_masts.station == name].iloc[0]
    mast_x, mast_y = tf_fwd.transform(mast.lon, mast.lat)
    ax.plot(mast_x, mast_y, marker="*", color="white", markersize=9,
            markeredgecolor="black", markeredgewidth=0.7, zorder=8, clip_on=True)

    # --- Inset -----------------------------------------------------------
    small_aoi = gpd.read_file(cfg["small_shp"]).to_crs(crs)
    sb        = small_aoi.total_bounds        # [xmin, ymin, xmax, ymax]
    pad_i     = 4 * max(res_x, res_y_abs)    # ~4 pixels of breathing room

    xi_min, yi_min = sb[0] - pad_i, sb[1] - pad_i
    xi_max, yi_max = sb[2] + pad_i, sb[3] + pad_i

    # Slice x/y to pixels that fall inside the inset extent
    x_sub = x[(x >= xi_min - res_x)     & (x <= xi_max + res_x)]
    y_sub = y[(y >= yi_min - res_y_abs)  & (y <= yi_max + res_y_abs)]

    inset_ax = ax.inset_axes(cfg["inset_loc"], zorder=10)
    inset_ax.set_facecolor("#cde2f0")

    # Basemap in inset (no extra pad — we want just what's in view)
    land_i, ice_i, rivers_i, valleys_i, lakes_i = _clip_basemap(
        crs, xi_min, yi_min, xi_max, yi_max, pad=0
    )
    land_i.plot(ax=inset_ax, color="#e8e0ce", edgecolor="#555555", linewidth=0.5, zorder=2)
    ice_i.plot( ax=inset_ax, color="#eef2ff", edgecolor="#9999cc", linewidth=0.3, zorder=3)
    if not valleys_i.empty:
        valleys_i.plot(ax=inset_ax, color="#b8d4e8", edgecolor="none",              zorder=4)
    if not lakes_i.empty:
        lakes_i.plot(  ax=inset_ax, color="#cde2f0", edgecolor="#6baed6", linewidth=0.4, zorder=5)
    if not rivers_i.empty:
        rivers_i.plot( ax=inset_ax, color="#4292c6", linewidth=0.7,                 zorder=6)

    # Pixel grid
    if len(x_sub) > 0 and len(y_sub) > 0:
        inset_ax.add_collection(
            _grid_line_collection(
                x_sub, y_sub, res_x, res_y_abs,
                colors="#444444", linewidths=0.6, alpha=0.65, zorder=4,
            )
        )

    # Proximity pixel: find the raster cell containing the mast and fill it
    prox = _proximity[_proximity.stid == name.lower()].iloc[0]
    prox_x, prox_y = tf_fwd.transform(prox.lon, prox.lat)
    ix = np.argmin(np.abs(x - prox_x))
    iy = np.argmin(np.abs(y - prox_y))
    inset_ax.add_patch(Rectangle(
        (x[ix] - res_x / 2, y[iy] - res_y_abs / 2),
        res_x, res_y_abs,
        facecolor="#fc8d59", alpha=0.55,
        edgecolor="#d73027", linewidth=1.5,
        zorder=5,
    ))

    # Main mast marker on the inset
    inset_ax.plot(mast_x, mast_y, marker="*", color="white", markersize=10,
                  markeredgecolor="black", markeredgewidth=0.7, zorder=8)

    inset_ax.set_xlim(xi_min, xi_max)
    inset_ax.set_ylim(yi_min, yi_max)
    inset_ax.set_aspect("equal")
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(1.4)

    # Drop shadow on the inset background patch — in display (pixel) units
    # so it's unaffected by projection or axes aspect ratio
    inset_ax.patch.set_path_effects([
        pe.withSimplePatchShadow(
            offset=(5, -5),
            shadow_rgbFace=(0.15, 0.15, 0.15),
            alpha=0.45,
        ),
        pe.Normal(),
    ])

    # Scale bar: 2 km
    sb_len = 2_000
    sb_x0  = xi_min + (xi_max - xi_min) * 0.06
    sb_y0  = yi_min + (yi_max - yi_min) * 0.14
    sb_h   = (yi_max - yi_min) * 0.025
    inset_ax.plot([sb_x0, sb_x0 + sb_len], [sb_y0, sb_y0],
                  color="black", lw=1.5, solid_capstyle="butt", zorder=7)
    for xv in [sb_x0, sb_x0 + sb_len]:
        inset_ax.plot([xv, xv], [sb_y0 - sb_h, sb_y0 + sb_h],
                      color="black", lw=1.0, zorder=7)
    inset_ax.text(sb_x0 + sb_len / 2, sb_y0 - sb_h * 1.8,
                  "2 km", ha="center", va="top", fontsize=5.5, zorder=7)

    # Connect inset to its location on the main map
    ax.indicate_inset_zoom(inset_ax, edgecolor="#333333", alpha=0.7, lw=0.8)

    for ext in ("png", "pdf"):
        outpath = outdir / f"site_map_{name.lower()}.{ext}"
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"Saved: {outpath}")
    plt.close(fig)
