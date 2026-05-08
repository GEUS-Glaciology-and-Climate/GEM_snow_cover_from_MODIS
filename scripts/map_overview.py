#!/usr/bin/env python3
"""Overview map of Greenland with GEM study site AOIs — UTM Zone 24N."""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from matplotlib.lines import Line2D
from shapely.geometry import LineString, MultiPoint, Point

workingdir = Path("/home/shl/mdrev/projects/modis/snow_cover")
SDFI_DIR   = Path("/home/shl/mdrev/data/sdfi/G250_VEKTOR")
CRS = "EPSG:32624"  # UTM Zone 24N

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dissolve(path, simplify_m=2000):
    """Load shapefile → reproject → repair → dissolve → simplify."""
    gdf = gpd.read_file(path).to_crs(CRS)
    gdf["geometry"] = gdf.buffer(0)
    dissolved = gdf.dissolve().simplify(simplify_m)
    return gpd.GeoDataFrame(geometry=dissolved, crs=CRS)


def _make_graticule(lats, lons, crs_to, lat_range=(57, 87), lon_range=(-75, 5), n=800):
    """Return dicts {value: LineString} for parallels and meridians in *crs_to*."""
    tf = pyproj.Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)
    parallels, meridians = {}, {}
    for lat in lats:
        pts = [tf.transform(lon, lat) for lon in np.linspace(*lon_range, n)]
        pts = [(x, y) for x, y in pts if np.isfinite(x) and np.isfinite(y)]
        if len(pts) > 1:
            parallels[lat] = LineString(pts)
    for lon in lons:
        pts = [tf.transform(lon, lat) for lat in np.linspace(*lat_range, n)]
        pts = [(x, y) for x, y in pts if np.isfinite(x) and np.isfinite(y)]
        if len(pts) > 1:
            meridians[lon] = LineString(pts)
    return parallels, meridians


def _border_intersections(line, border):
    """Return list of (x, y) where *line* crosses *border* LineString."""
    isect = line.intersection(border)
    if isect.is_empty:
        return []
    if isinstance(isect, Point):
        return [(isect.x, isect.y)]
    if isinstance(isect, MultiPoint):
        return [(p.x, p.y) for p in isect.geoms]
    # LineString overlap — unlikely but take first coord
    if hasattr(isect, "coords"):
        return [isect.coords[0]]
    return []


def _fmt_lon(v):
    if v < 0:
        return f"{int(abs(v))}°W"
    return f"{int(v)}°E" if v else "0°"


def _fmt_lat(v):
    return f"{int(v)}°N"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
land = _load_dissolve(SDFI_DIR / "Land.shp")
ice  = _load_dissolve(SDFI_DIR / "Is.shp")

site_cfg = {
    "Zackenberg": {
        "file":   "zackenberg_aoi.shp",
        "color":  "#d73027",
        "label_offset": (-220_000, 140_000),   # (Δx m, Δy m) from centroid
    },
    "Disko": {
        "file":   "disko_aoi.shp",
        "color":  "#4575b4",
        "label_offset": (-280_000, 80_000),
    },
    "Nuuk": {
        "file":   "nuuk_aoi.shp",
        "color":  "#1a9641",
        "label_offset": (-240_000, -80_000),
    },
}
aois = {
    name: (gpd.read_file(workingdir / "shp" / cfg["file"]).to_crs(CRS), cfg)
    for name, cfg in site_cfg.items()
}

# ---------------------------------------------------------------------------
# Map extent
# ---------------------------------------------------------------------------
xmin_d, ymin_d, xmax_d, ymax_d = land.total_bounds
pad_x = (xmax_d - xmin_d) * 0.04
pad_y = (ymax_d - ymin_d) * 0.03
xmin, xmax = xmin_d - pad_x, xmax_d + pad_x
ymin, ymax = ymin_d - pad_y, ymax_d + pad_y

# ---------------------------------------------------------------------------
# Graticule
# ---------------------------------------------------------------------------
LATS = range(60, 90, 10)
LONS = range(-60, -20, 10)  # those visible within the map extent
parallels, meridians = _make_graticule(LATS, LONS, CRS)

# Border lines used to find graticule–frame intersections
eps = 5_000   # m — extend border slightly beyond corners
left_bdr   = LineString([(xmin, ymin - eps), (xmin, ymax + eps)])
right_bdr  = LineString([(xmax, ymin - eps), (xmax, ymax + eps)])
bottom_bdr = LineString([(xmin - eps, ymin), (xmax + eps, ymin)])
top_bdr    = LineString([(xmin - eps, ymax), (xmax + eps, ymax)])

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 9.5))
ax.set_facecolor("#cde2f0")   # ocean

land.plot(ax=ax, color="#e8e0ce", edgecolor="#555555", linewidth=0.3, zorder=2)
ice.plot( ax=ax, color="#eef2ff", edgecolor="#9999cc", linewidth=0.2, zorder=3)

# Graticule lines
for line in parallels.values():
    xy = np.array(line.coords)
    ax.plot(xy[:, 0], xy[:, 1], "--", color="#888888", lw=0.35, alpha=0.55, zorder=1)
for line in meridians.values():
    xy = np.array(line.coords)
    ax.plot(xy[:, 0], xy[:, 1], "--", color="#888888", lw=0.35, alpha=0.55, zorder=1)

# AOI boxes and labels
legend_handles = []
for name, (gdf, cfg) in aois.items():
    color = cfg["color"]
    dx, dy = cfg["label_offset"]
    gdf.plot(ax=ax, facecolor="none", edgecolor=color, linewidth=2.5, zorder=5)
    centroid = gdf.dissolve().centroid.iloc[0]
    cx, cy = centroid.x, centroid.y
    ax.annotate(
        name,
        xy=(cx, cy),
        xytext=(cx + dx, cy + dy),
        fontsize=9,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=10),
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.88,
            edgecolor=color,
            linewidth=0.8,
        ),
        zorder=7,
    )
    legend_handles.append(Line2D([0], [0], color=color, lw=2.5, label=name))

# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

# ---------------------------------------------------------------------------
# Graticule labels — placed outside the frame at border intersections
# Tick mark length and label offset are in data units (metres)
# ---------------------------------------------------------------------------
TICK_LEN   = 25_000   # inward tick length (m)
LAB_OFFSET = 35_000   # label distance from border (m)
FONT_KW = dict(fontsize=7.5, clip_on=False)

# Parallels → left and right borders
for lat, line in parallels.items():
    for bdr, side in [(left_bdr, "left"), (right_bdr, "right")]:
        pts = _border_intersections(line, bdr)
        for px, py in pts:
            if not (ymin <= py <= ymax):
                continue
            if side == "left":
                ax.plot([px, px + TICK_LEN], [py, py], color="black", lw=0.8, zorder=6, clip_on=False)
                ax.text(px - LAB_OFFSET, py, _fmt_lat(lat),
                        ha="right", va="center", **FONT_KW)
            else:
                ax.plot([px - TICK_LEN, px], [py, py], color="black", lw=0.8, zorder=6, clip_on=False)
                ax.text(px + LAB_OFFSET, py, _fmt_lat(lat),
                        ha="left", va="center", **FONT_KW)

# Meridians → bottom and top borders
for lon, line in meridians.items():
    for bdr, side in [(bottom_bdr, "bottom"), (top_bdr, "top")]:
        pts = _border_intersections(line, bdr)
        for px, py in pts:
            if not (xmin <= px <= xmax):
                continue
            if side == "bottom":
                ax.plot([px, px], [py, py + TICK_LEN], color="black", lw=0.8, zorder=6, clip_on=False)
                ax.text(px, py - LAB_OFFSET, _fmt_lon(lon),
                        ha="center", va="top", rotation=45, **FONT_KW)
            else:
                ax.plot([px, px], [py - TICK_LEN, py], color="black", lw=0.8, zorder=6, clip_on=False)
                ax.text(px, py + LAB_OFFSET, _fmt_lon(lon),
                        ha="center", va="bottom", rotation=45, **FONT_KW)

# ---------------------------------------------------------------------------
# Scale bar (500 km, bottom-left corner inside map)
# ---------------------------------------------------------------------------
sb_x0 = xmin + 60_000
sb_y0 = ymin + 120_000
sb_len = 500_000   # 500 km in metres

ax.plot([sb_x0, sb_x0 + sb_len], [sb_y0, sb_y0], color="black", lw=2.0, zorder=6)
for x in [sb_x0, sb_x0 + sb_len]:
    ax.plot([x, x], [sb_y0 - 18_000, sb_y0 + 18_000], color="black", lw=1.5, zorder=6)
ax.text(sb_x0 + sb_len / 2, sb_y0 - 35_000, "500 km",
        ha="center", va="top", fontsize=7, zorder=6)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
ax.legend(
    handles=legend_handles,
    loc="lower right",
    fontsize=8,
    title="Study sites",
    title_fontsize=8,
    framealpha=0.9,
    edgecolor="#aaaaaa",
    fancybox=False,
)

plt.tight_layout()

outdir = workingdir / "figures"
outdir.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    outpath = outdir / f"greenland_overview_map.{ext}"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved: {outpath}")
