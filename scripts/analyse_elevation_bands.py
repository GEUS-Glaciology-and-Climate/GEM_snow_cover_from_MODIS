"""
Snow-Free Days by Elevation Band
=================================
Reprojects ArcticDEM elevation to the MODIS 500m grid, bins pixels into
100 m elevation bands, and computes mean annual snow-free days per band.

Snow-free is defined as NDSI < 40% (calibrated threshold).

Outputs:
  results/csvs/{site}/sfd_elevation_bands_{source}.csv
  figures/{site}/sfd_elevation_bands_{source}_heatmap.png
  figures/{site}/sfd_elevation_bands_{source}_lines.png
  figures/{site}/sfd_elevation_bands_comparison_lines.png  (if both sources present)
  figures/{site}/sfd_elevation_bands_comparison_heatmap.png (if both sources present)

Usage:
    python scripts/analyse_elevation_bands.py --site zackenberg --source masked
    python scripts/analyse_elevation_bands.py --site zackenberg --source filled
"""

import argparse
import yaml
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

THRESHOLD    = 40
BIN_WIDTH    = 100
MIN_PIXELS   = 5    # minimum valid pixels per bin to include in output
ELEV_MAX_PLOT = 1500  # upper elevation limit for line plots (m)

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True)
parser.add_argument("--source", choices=["masked", "filled"], default="masked")
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site   = cfg["site"]
source = args.source

if source == "masked":
    nc_dir   = Path(f"netcdf/{site}_masked/")
    nc_var   = "snow_cover_fraction_masked"
else:
    nc_dir   = Path(f"netcdf/{site}_filled/")
    nc_var   = "scf_filled"

out_dir = Path(f"figures/{site}/")
csv_dir = Path(f"results/csvs/{site}/")
out_dir.mkdir(parents=True, exist_ok=True)
csv_dir.mkdir(parents=True, exist_ok=True)

# --- Load SCF data ---
nc_files = sorted(nc_dir.glob(f"{site}_scf_*.nc"))
if not nc_files:
    raise FileNotFoundError(f"No NetCDF files found in {nc_dir}")

print(f"Loading {len(nc_files)} annual SCF files ({source})...")
ds = xr.open_mfdataset(nc_files, combine="nested", concat_dim="time", chunks={"time": 30})
da = ds[nc_var]
if source == "masked":
    da = da.where(da <= 100)

# --- Load terrain and reproject elevation to MODIS grid ---
terrain_path = Path(f"netcdf/{site}/{site}_terrain.nc")
if not terrain_path.exists():
    raise FileNotFoundError(f"Terrain file not found: {terrain_path}. Run make terrain first.")

print("Reprojecting elevation to MODIS grid...")
terrain = xr.open_dataset(terrain_path)
elev_32m = terrain["elevation"].rio.write_crs(cfg["target_epsg"])

# reproject_match reprojects and resamples to exactly the MODIS pixel grid
# averaging the 32m DEM pixels that fall within each 500m MODIS pixel
da_template = da.isel(time=0).drop_vars("time").rio.write_crs(cfg["target_epsg"])
elev_modis = elev_32m.rio.reproject_match(da_template)

elev_vals = elev_modis.values  # 2D array, NaN where ocean/no data

valid_elev = elev_vals[np.isfinite(elev_vals)]
elev_min = np.floor(valid_elev.min() / BIN_WIDTH) * BIN_WIDTH
elev_max = np.ceil(valid_elev.max()  / BIN_WIDTH) * BIN_WIDTH
bin_edges = np.arange(elev_min, elev_max + BIN_WIDTH, BIN_WIDTH)
bin_labels = bin_edges[:-1].astype(int)   # lower edge of each bin

print(f"Elevation range: {elev_min:.0f} – {elev_max:.0f} m")
print(f"Number of {BIN_WIDTH} m bins: {len(bin_labels)}")

# Assign each pixel to a bin index (-1 = outside range / NaN)
bin_idx = np.full(elev_vals.shape, -1, dtype=int)
valid_mask = np.isfinite(elev_vals)
bin_idx[valid_mask] = np.digitize(elev_vals[valid_mask], bin_edges) - 1
# Clip to valid range (top edge pixels land in bin == len(bin_labels))
bin_idx = np.clip(bin_idx, -1, len(bin_labels) - 1)

# --- Compute annual SFD per elevation bin ---
years = np.unique(da.time.dt.year.values)
records = []

for year in years:
    yearly = da.sel(time=da.time.dt.year == year)
    if len(yearly.time) < 320:
        print(f"  {year}: skipped ({len(yearly.time)} days)")
        continue

    sfd_map = (yearly < THRESHOLD).sum(dim="time").values.astype(float)
    # NaN where all days were invalid
    sfd_map[~np.isfinite(sfd_map)] = np.nan

    for b, label in enumerate(bin_labels):
        mask = bin_idx == b
        sfd_in_bin = sfd_map[mask]
        n_valid = np.sum(np.isfinite(sfd_in_bin))
        if n_valid < MIN_PIXELS:
            continue
        records.append({
            "year":       year,
            "elev_lower": label,
            "n_pixels":   int(n_valid),
            "sfd_mean":   float(np.nanmean(sfd_in_bin)),
            "sfd_std":    float(np.nanstd(sfd_in_bin)),
        })

    print(f"  {year}: done")

df = pd.DataFrame(records)

csv_path = csv_dir / f"sfd_elevation_bands_{source}.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved CSV: {csv_path}")

# --- Pivot to year × elevation matrix for heatmap ---
pivot = df.pivot(index="elev_lower", columns="year", values="sfd_mean")
pivot = pivot.sort_index()

# ============================================================
# Helpers
# ============================================================
def plot_heatmap(piv, title, out_path):
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(
        piv.values, aspect="auto", origin="lower",
        extent=[
            piv.columns.min() - 0.5, piv.columns.max() + 0.5,
            piv.index.min() - BIN_WIDTH / 2, piv.index.max() + BIN_WIDTH / 2,
        ],
        cmap="YlOrRd", vmin=0,
    )
    fig.colorbar(im, ax=ax, label="Mean snow-free days per year")
    ax.set_xlabel("Year"); ax.set_ylabel("Elevation (m a.s.l.)")
    ax.set_title(title)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap: {out_path.name}")

def clip_pivot(piv):
    """Clip pivot to second-lowest bin through ELEV_MAX_PLOT."""
    elev_min_clip = piv.index[1] if len(piv.index) > 1 else piv.index[0]
    return piv.loc[(piv.index >= elev_min_clip) & (piv.index <= ELEV_MAX_PLOT)]

def fit_slopes(piv):
    """Return per-year linear slope (SFD per 100 m) using OLS on clipped data."""
    from numpy.polynomial import polynomial as P
    slopes = {}
    for year in piv.columns:
        col = piv[year].dropna()
        if len(col) < 3:
            continue
        elev_c = col.index + BIN_WIDTH / 2
        # slope in days per metre → convert to days per 100 m
        coeffs = np.polyfit(elev_c, col.values, 1)
        slopes[year] = coeffs[0] * 100   # days per 100 m
    return slopes

def add_lines(piv, ax, cmap_name="viridis", add_slope=True):
    """Plot per-year SFD profiles and overlay mean linear slope."""
    piv = clip_pivot(piv)
    years_avail = piv.columns.tolist()
    cmap2  = plt.cm.get_cmap(cmap_name)
    colors = cmap2(np.linspace(0.0, 1.0, len(years_avail)))

    for year, color in zip(years_avail, colors):
        col = piv[year].dropna()
        if len(col) < 3:
            continue
        ax.plot(col.values, col.index + BIN_WIDTH / 2,
                marker="o", markersize=2.5, linewidth=1.2, color=color)

    sm = plt.cm.ScalarMappable(
        cmap=cmap2,
        norm=plt.Normalize(vmin=min(years_avail), vmax=max(years_avail)))
    sm.set_array([])
    ax.get_figure().colorbar(sm, ax=ax, label="Year", shrink=0.8)

    if add_slope:
        slopes = fit_slopes(piv)
        if slopes:
            slope_vals = np.array(list(slopes.values()))
            mean_slope = np.mean(slope_vals)
            std_slope  = np.std(slope_vals)

            # Draw mean regression line across the clipped elevation range
            elev_centers = piv.index + BIN_WIDTH / 2
            # Use mean of all valid per-year fits to anchor the line
            all_sfd, all_elev = [], []
            for year in piv.columns:
                col = piv[year].dropna()
                if len(col) < 3:
                    continue
                all_sfd.extend(col.values)
                all_elev.extend(col.index + BIN_WIDTH / 2)
            if len(all_elev) > 3:
                coeffs = np.polyfit(all_elev, all_sfd, 1)
                fit_sfd = np.polyval(coeffs, elev_centers)
                ax.plot(fit_sfd, elev_centers, "k--", linewidth=1.8,
                        label=f"Mean slope: {mean_slope:+.1f} d / 100 m "
                              f"(±{std_slope:.1f})")
                ax.legend(fontsize=8, loc="lower right")

def plot_lines(piv, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 8))
    add_lines(piv, ax)
    piv_c = clip_pivot(piv)
    ax.set_xlabel("Mean snow-free days per year", fontsize=9)
    ax.set_ylabel("Elevation (m a.s.l.)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(piv_c.index[0], ELEV_MAX_PLOT + BIN_WIDTH)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved line plot: {out_path.name}")

# ============================================================
# Plot 1: Heatmap
# ============================================================
plot_heatmap(
    pivot,
    title=f"{site.capitalize()}: Snow-free days by elevation band ({source}, NDSI < {THRESHOLD}%)",
    out_path=out_dir / f"sfd_elevation_bands_{source}_heatmap.png",
)

# ============================================================
# Plot 2: Line plot
# ============================================================
plot_lines(
    pivot,
    title=f"{site.capitalize()}: SFD by elevation ({source}, NDSI < {THRESHOLD}%)",
    out_path=out_dir / f"sfd_elevation_bands_{source}_lines.png",
)

# ============================================================
# Plot 3: Comparison — masked vs filled (if both CSVs exist)
# ============================================================
csv_masked = csv_dir / "sfd_elevation_bands_masked.csv"
csv_filled = csv_dir / "sfd_elevation_bands_filled.csv"

if csv_masked.exists() and csv_filled.exists():
    print("\nBoth sources found — generating comparison plots...")
    piv_m = (pd.read_csv(csv_masked)
             .pivot(index="elev_lower", columns="year", values="sfd_mean")
             .sort_index())
    piv_f = (pd.read_csv(csv_filled)
             .pivot(index="elev_lower", columns="year", values="sfd_mean")
             .sort_index())

    # Side-by-side line plot with elevation clipping and slopes
    elev_bottom = max(clip_pivot(piv_m).index[0], clip_pivot(piv_f).index[0])
    fig_c, axes_c = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    add_lines(piv_m, axes_c[0])
    axes_c[0].set_xlabel("Mean snow-free days per year", fontsize=9)
    axes_c[0].set_ylabel("Elevation (m a.s.l.)", fontsize=9)
    axes_c[0].set_title("Masked (land only)", fontsize=10)
    axes_c[0].set_ylim(elev_bottom, ELEV_MAX_PLOT + BIN_WIDTH)
    axes_c[0].yaxis.set_major_locator(ticker.MultipleLocator(200))
    axes_c[0].yaxis.set_minor_locator(ticker.MultipleLocator(100))
    axes_c[0].grid(True, alpha=0.3)

    add_lines(piv_f, axes_c[1])
    axes_c[1].set_xlabel("Mean snow-free days per year", fontsize=9)
    axes_c[1].set_title("Gap-filled (land + glacier)", fontsize=10)
    axes_c[1].set_ylim(elev_bottom, ELEV_MAX_PLOT + BIN_WIDTH)
    axes_c[1].yaxis.set_major_locator(ticker.MultipleLocator(200))
    axes_c[1].yaxis.set_minor_locator(ticker.MultipleLocator(100))
    axes_c[1].grid(True, alpha=0.3)

    fig_c.suptitle(
        f"{site.capitalize()}: Snow-free days by elevation band (NDSI < {THRESHOLD}%)",
        fontsize=11)
    fig_c.tight_layout()
    fig_c.savefig(out_dir / "sfd_elevation_bands_comparison_lines.png", dpi=150)
    plt.close(fig_c)
    print("Saved comparison line plot")

    # Difference heatmap (filled − masked)
    common_years = piv_m.columns.intersection(piv_f.columns)
    common_elev  = piv_m.index.intersection(piv_f.index)
    diff = piv_f.loc[common_elev, common_years] - piv_m.loc[common_elev, common_years]

    fig_d, ax_d = plt.subplots(figsize=(14, 7))
    vmax = np.nanpercentile(np.abs(diff.values[np.isfinite(diff.values)]), 95)
    im_d = ax_d.imshow(
        diff.values, aspect="auto", origin="lower",
        extent=[
            diff.columns.min() - 0.5, diff.columns.max() + 0.5,
            diff.index.min() - BIN_WIDTH / 2, diff.index.max() + BIN_WIDTH / 2,
        ],
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    fig_d.colorbar(im_d, ax=ax_d, label="ΔSFD filled − masked (days)")
    ax_d.set_xlabel("Year"); ax_d.set_ylabel("Elevation (m a.s.l.)")
    ax_d.set_title(
        f"{site.capitalize()}: Difference in SFD — gap-filled minus masked (NDSI < {THRESHOLD}%)")
    ax_d.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_d.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax_d.yaxis.set_major_locator(ticker.MultipleLocator(200))
    plt.setp(ax_d.get_xticklabels(), rotation=45, ha="right")
    fig_d.tight_layout()
    fig_d.savefig(out_dir / "sfd_elevation_bands_comparison_heatmap.png", dpi=150)
    plt.close(fig_d)
    print("Saved difference heatmap")
