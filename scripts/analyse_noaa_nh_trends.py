"""
NOAA Northern Hemisphere Snow Cover Extent — Trend Analysis
===========================================================
Two analyses using the NOAA NH SCE CDR (weekly binary, ~190 km grid):

  Analysis 1 — Zackenberg pixel: snow-free days derived from the nearest
    grid cell. Compared with MODIS-derived snow-free days (NDSI < 40%).
    Trends tested for: full record (1967–2025) and MODIS period (2001–2025).

  Analysis 2 — NH total snow cover extent: total snow-covered land area
    (million km²) summed over all land pixels and aggregated to annual means.
    Trends tested for: full record (1967–2025) and MODIS period (2000–2025).

Usage:
    python scripts/analyse_noaa_nh_trends.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymannkendall as mk
import xarray as xr
from pathlib import Path

# --- Paths ---
workingdir = Path("/home/shl/mdrev/projects/modis/snow_cover")
# Local copy of the NOAA NH SCE CDR (downloaded from NCEI).
# The dataset is also accessible via OPeNDAP, but the filename encodes the
# end date and changes with each update, making the URL fragile:
#   https://www.ncei.noaa.gov/thredds/dodsC/cdr/snowcover/nhsce_v01r01_19661004_20260601.nc
NOAA_PATH = Path("/home/shl/mdrev/data/noaa/netcdfs/full/nhsce_v01r01_19661004_20260601.nc")
modis_csv  = workingdir / "results/csvs/zackenberg/snow_free_days.csv"
fig_dir    = workingdir / "figures/zackenberg/"
csv_dir    = workingdir / "results/csvs/zackenberg/"
fig_dir.mkdir(parents=True, exist_ok=True)
csv_dir.mkdir(parents=True, exist_ok=True)

alpha       = 0.05
threshold   = 40   # MODIS NDSI threshold for comparison
modis_start = 2001

# Zackenberg: polar stereographic grid coordinates (km)
x0, y0 = 1621, -1060

# =============================================================
# 1. Load and process NOAA data
# =============================================================
ds = xr.open_dataset(NOAA_PATH)

sce_ts = ds.snow_cover_extent.sel(x=x0, y=y0, method="nearest")
lat = float(ds.latitude.sel(x=x0, y=y0, method="nearest").values)
lon = float(ds.longitude.sel(x=x0, y=y0, method="nearest").values)
print(f"Nearest NOAA grid cell: lat={lat:.2f}°N, lon={lon:.2f}°E")

# Mask invalid fill values (valid range is 0–1)
sce_ts = sce_ts.where(sce_ts >= 0)

# Forward-fill weekly observations to daily, then sum snow-covered days per year
snow_covered = sce_ts.to_pandas().resample("D").ffill().resample("YE").sum()
snow_free_noaa = (365 - snow_covered).rename("snow_free_days")
snow_free_noaa.index = snow_free_noaa.index.year
snow_free_noaa.index.name = "year"

# Drop first partial year (1966) and any incomplete trailing year
snow_free_noaa = snow_free_noaa.iloc[1:]
snow_free_noaa = snow_free_noaa[snow_free_noaa.index <= 2025]
snow_free_noaa = snow_free_noaa.dropna()

snow_free_noaa.to_csv(csv_dir / "noaa_zackenberg_snow_free_days.csv")
print(f"NOAA series: {snow_free_noaa.index[0]}–{snow_free_noaa.index[-1]}, "
      f"n={len(snow_free_noaa)}")

# =============================================================
# 2. Load MODIS data
# =============================================================
modis_df  = pd.read_csv(modis_csv, index_col="year")
modis_sfd = modis_df[f"sfd_lt{threshold}"].dropna()
print(f"MODIS series: {modis_sfd.index[0]}–{modis_sfd.index[-1]}, "
      f"n={len(modis_sfd)}")

# =============================================================
# 3. Mann-Kendall trend tests
# =============================================================
noaa_full    = snow_free_noaa
noaa_overlap = snow_free_noaa.loc[modis_start:]

mk_noaa_full    = mk.original_test(noaa_full.values,    alpha=alpha)
mk_noaa_overlap = mk.original_test(noaa_overlap.values, alpha=alpha)
mk_modis        = mk.original_test(modis_sfd.values,    alpha=alpha)

print("\n" + "=" * 65)
print("Mann-Kendall trend results")
print("=" * 65)
tests = [
    ("NOAA full (1967–2025)",           mk_noaa_full,    len(noaa_full)),
    (f"NOAA MODIS period ({modis_start}–2025)", mk_noaa_overlap, len(noaa_overlap)),
    (f"MODIS ({modis_start}–2025)",      mk_modis,        len(modis_sfd)),
]
print(f"{'Dataset':<35} {'Slope (d/yr)':>12} {'p-value':>9} {'Significant':>12}")
print("-" * 70)
for label, result, _ in tests:
    sig = "Yes" if result.p < alpha else "No"
    print(f"{label:<35} {result.slope:>12.3f} {result.p:>9.4f} {sig:>12}")


def _trend_line(result, n):
    """Sen's slope line using 0-indexed positions (consistent with project convention)."""
    return result.intercept + result.slope * np.arange(n)


# =============================================================
# 4. Figure
# =============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

# ── Panel 1: Full NOAA record ─────────────────────────────────
years_full = noaa_full.index.values
ax1.fill_betweenx(
    [0, 400],
    modis_start - 0.5, years_full[-1] + 0.5,
    color="gray", alpha=0.07, zorder=0, label=f"MODIS period ({modis_start}–{years_full[-1]})"
)
ax1.plot(years_full, noaa_full.values, "o-", color="steelblue", linewidth=1.5,
         markersize=4, label="NOAA NH SCE (Zackenberg ~190 km cell)")

# Full-record trend
tl_full = _trend_line(mk_noaa_full, len(years_full))
if mk_noaa_full.p < alpha:
    ax1.plot(years_full, tl_full, "-", color="steelblue", linewidth=2.5, alpha=0.75,
             label=(f"Full-record trend: {mk_noaa_full.slope:+.2f} d yr$^{{-1}}$ "
                    f"(p = {mk_noaa_full.p:.3f})"))
else:
    ax1.plot(years_full, tl_full, "-", color="steelblue", linewidth=2, alpha=0.4,
             label=(f"Full-record trend: {mk_noaa_full.slope:+.2f} d yr$^{{-1}}$ "
                    f"(p = {mk_noaa_full.p:.3f}, n.s.)"))

# MODIS-period trend on NOAA
years_ov = noaa_overlap.index.values
tl_ov    = _trend_line(mk_noaa_overlap, len(years_ov))
if mk_noaa_overlap.p < alpha:
    ax1.plot(years_ov, tl_ov, "--", color="firebrick", linewidth=2.5, alpha=0.8,
             label=(f"MODIS-period trend: {mk_noaa_overlap.slope:+.2f} d yr$^{{-1}}$ "
                    f"(p = {mk_noaa_overlap.p:.3f})"))
else:
    ax1.plot(years_ov, tl_ov, "--", color="firebrick", linewidth=2, alpha=0.45,
             label=(f"MODIS-period trend: {mk_noaa_overlap.slope:+.2f} d yr$^{{-1}}$ "
                    f"(p = {mk_noaa_overlap.p:.3f}, n.s.)"))

ax1.set_ylabel("Snow-free days")
ax1.set_title("NOAA NH Snow Cover Extent — Zackenberg grid cell (1967–2025)")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(years_full[0] - 0.5, years_full[-1] + 0.5)
ax1.set_ylim(bottom=0)

# ── Panel 2: MODIS period — NOAA vs MODIS (dual y-axis) ───────
years_mod = modis_sfd.index.values
tl_mod    = _trend_line(mk_modis, len(years_mod))

ax2r = ax2.twinx()

# NOAA on left axis
ax2.plot(years_ov, noaa_overlap.values, "o-", color="steelblue", linewidth=1.5,
         markersize=5, label="NOAA NH SCE")
if mk_noaa_overlap.p < alpha:
    ax2.plot(years_ov, tl_ov, "-", color="steelblue", linewidth=2.5, alpha=0.75,
             label=(f"NOAA trend: {mk_noaa_overlap.slope:+.2f} d yr$^{{-1}}$ "
                    f"(p = {mk_noaa_overlap.p:.3f})"))
else:
    ax2.plot(years_ov, tl_ov, "-", color="steelblue", linewidth=2, alpha=0.4,
             label=(f"NOAA trend: {mk_noaa_overlap.slope:+.2f} d yr$^{{-1}}$ "
                    f"(p = {mk_noaa_overlap.p:.3f}, n.s.)"))

# MODIS on right axis
ax2r.plot(years_mod, modis_sfd.values, "s--", color="darkorange", linewidth=1.5,
          markersize=5, label=f"MODIS (NDSI $<$ {threshold}\\%)")
if mk_modis.p < alpha:
    ax2r.plot(years_mod, tl_mod, "--", color="darkorange", linewidth=2.5, alpha=0.75,
              label=(f"MODIS trend: {mk_modis.slope:+.2f} d yr$^{{-1}}$ "
                     f"(p = {mk_modis.p:.3f})"))
else:
    ax2r.plot(years_mod, tl_mod, "--", color="darkorange", linewidth=2, alpha=0.4,
              label=(f"MODIS trend: {mk_modis.slope:+.2f} d yr$^{{-1}}$ "
                     f"(p = {mk_modis.p:.3f}, n.s.)"))

ax2.set_xlabel("Year")
ax2.set_ylabel("Snow-free days (NOAA)", color="steelblue")
ax2r.set_ylabel(f"Snow-free days (MODIS, NDSI $<$ {threshold}%)", color="darkorange")
ax2.tick_params(axis="y", labelcolor="steelblue")
ax2r.tick_params(axis="y", labelcolor="darkorange")
ax2.set_title(
    f"NOAA NH SCE vs MODIS Snow-Free Days — Zackenberg ({modis_start}–{years_ov[-1]})"
)
ax2.set_xlim(modis_start - 0.5, years_ov[-1] + 0.5)
ax2.grid(True, alpha=0.3)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

fig.tight_layout()
out_png = fig_dir / "noaa_nh_snow_cover_trends.png"
fig.savefig(out_png, dpi=150)
print(f"\nSaved: {out_png}")

# =============================================================
# 5. LaTeX summary table
# =============================================================
latex_dir = workingdir / "results/latex/zackenberg/"
latex_dir.mkdir(parents=True, exist_ok=True)
tex_path = latex_dir / "noaa_trend_summary.tex"

with open(tex_path, "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Mann-Kendall trend analysis of snow-free days at Zackenberg. "
            "NOAA NH SCE CDR uses a $\\sim$190\\,km polar stereographic grid cell; "
            "MODIS uses NDSI $<$ 40\\%.}\n")
    f.write("\\label{tab:noaa_trend_summary}\n")
    f.write("\\begin{tabular}{lrrrr}\n")
    f.write("\\hline\n")
    f.write("Dataset & Period & $n$ & Sen's slope (d\\,yr$^{-1}$) & $p$-value \\\\\n")
    f.write("\\hline\n")
    rows = [
        ("NOAA NH SCE",  "1967--2025",             len(noaa_full),    mk_noaa_full),
        ("NOAA NH SCE",  f"{modis_start}--2025",   len(noaa_overlap), mk_noaa_overlap),
        (f"MODIS (NDSI $<$ {threshold}\\%)",
                         f"{modis_start}--2025",   len(modis_sfd),    mk_modis),
    ]
    for name, period, n, result in rows:
        sig = "" if result.p < alpha else " (n.s.)"
        f.write(f"{name} & {period} & {n} & "
                f"{result.slope:+.3f}{sig} & {result.p:.4f} \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"Saved LaTeX table: {tex_path}")

# =============================================================
# 6. NH total snow cover extent — annual means
# =============================================================
print("\n" + "=" * 65)
print("Analysis 2: NH total snow cover extent (annual means)")
print("=" * 65)

nh_start_modis = 2000  # MODIS era start for this analysis

# Total snow-covered area per week (million km²), land pixels only
land_mask = ds.land.values == 1
area_land  = ds.area.values[land_mask]   # shape (n_land,)

sce_flat = ds.snow_cover_extent.values   # (time, y, x)
n_time   = sce_flat.shape[0]
sce_land = sce_flat.reshape(n_time, -1)[:, land_mask.ravel()]  # (time, n_land)

# Mask invalid values (valid range 0–1)
sce_land = sce_land.astype(float)
sce_land[sce_land < 0] = np.nan

# Weighted sum → total snow area in million km²
total_snow_area = np.nansum(sce_land * area_land[np.newaxis, :], axis=1) / 1e6

# Build weekly time series
ts_weekly = pd.Series(total_snow_area, index=pd.DatetimeIndex(ds.time.values))

# Annual means (calendar year), drop first and last partial years
ts_annual = ts_weekly.resample("YE").mean()
ts_annual.index = ts_annual.index.year
ts_annual.index.name = "year"
ts_annual = ts_annual[(ts_annual.index >= 1967) & (ts_annual.index <= 2025)]
ts_annual = ts_annual.dropna()

ts_annual.rename("snow_extent_mkm2").to_csv(
    csv_dir.parent / "noaa_nh_snow_extent_annual.csv", header=True
)
print(f"NH extent series: {ts_annual.index[0]}–{ts_annual.index[-1]}, "
      f"n={len(ts_annual)}")
print(f"Mean NH snow extent: {ts_annual.mean():.2f} M km²  "
      f"(range {ts_annual.min():.2f}–{ts_annual.max():.2f})")

# Mann-Kendall tests
nh_full    = ts_annual
nh_overlap = ts_annual.loc[nh_start_modis:]

mk_nh_full    = mk.original_test(nh_full.values,    alpha=alpha)
mk_nh_overlap = mk.original_test(nh_overlap.values, alpha=alpha)

print(f"\n{'Dataset':<35} {'Slope (M km²/yr)':>17} {'p-value':>9} {'Significant':>12}")
print("-" * 75)
for label, result in [
    ("NH extent full (1967–2025)",          mk_nh_full),
    (f"NH extent MODIS period ({nh_start_modis}–2025)", mk_nh_overlap),
]:
    sig = "Yes" if result.p < alpha else "No"
    print(f"{label:<35} {result.slope:>17.4f} {result.p:>9.4f} {sig:>12}")

# =============================================================
# 7. Subdomains: North America/Greenland and Eurasia/Asia
# =============================================================
subdomains = [
    ("North America & Greenland", -180,  0),
    ("Eurasia & Asia",               0, 180),
]

lon_2d    = ds.longitude.values  # (y, x)
sce_flat2 = sce_flat.reshape(n_time, -1)

sub_results = []  # list of dicts, one per subdomain

for sd_name, sd_lon_min, sd_lon_max in subdomains:
    print("\n" + "=" * 65)
    print(f"Analysis — {sd_name} (lon {sd_lon_min}° to {sd_lon_max}°)")
    print("=" * 65)

    sub_mask = land_mask & (lon_2d >= sd_lon_min) & (lon_2d <= sd_lon_max)
    area_sub = ds.area.values[sub_mask]
    sce_sub  = sce_flat2[:, sub_mask.ravel()].astype(float)
    sce_sub[sce_sub < 0] = np.nan

    total_snow_sub = np.nansum(sce_sub * area_sub[np.newaxis, :], axis=1) / 1e6
    ts_weekly = pd.Series(total_snow_sub, index=pd.DatetimeIndex(ds.time.values))
    ts_annual = ts_weekly.resample("YE").mean()
    ts_annual.index = ts_annual.index.year
    ts_annual.index.name = "year"
    ts_annual = ts_annual[(ts_annual.index >= 1967) & (ts_annual.index <= 2025)].dropna()

    slug = sd_name.lower().replace(" ", "_").replace("&", "and").replace("/", "_")
    ts_annual.rename("snow_extent_mkm2").to_csv(
        csv_dir.parent / f"noaa_{slug}_snow_extent_annual.csv", header=True
    )

    print(f"Land pixels: {sub_mask.sum()}  (total area {area_sub.sum()/1e6:.1f} M km²)")
    print(f"Series: {ts_annual.index[0]}–{ts_annual.index[-1]}, n={len(ts_annual)}")
    print(f"Mean: {ts_annual.mean():.2f} M km²  "
          f"(range {ts_annual.min():.2f}–{ts_annual.max():.2f})")

    sd_full    = ts_annual
    sd_overlap = ts_annual.loc[nh_start_modis:]
    mk_full    = mk.original_test(sd_full.values,    alpha=alpha)
    mk_overlap = mk.original_test(sd_overlap.values, alpha=alpha)

    print(f"\n  {'Period':<30} {'Slope (M km²/yr)':>17} {'p-value':>9} {'Sig':>5}")
    print("  " + "-" * 65)
    for label, result in [("Full (1967–2025)", mk_full),
                           (f"MODIS ({nh_start_modis}–2025)", mk_overlap)]:
        sig = "Yes" if result.p < alpha else "No"
        print(f"  {label:<30} {result.slope:>17.4f} {result.p:>9.4f} {sig:>5}")

    sub_results.append(dict(
        name=sd_name, lon_min=sd_lon_min, lon_max=sd_lon_max,
        full=sd_full, overlap=sd_overlap,
        mk_full=mk_full, mk_overlap=mk_overlap,
    ))

# =============================================================
# 8. Three-panel figure
# =============================================================
def _plot_extent_panel(ax, years_full, ts_full, mk_full, years_ov, ts_ov, mk_ov,
                       color, series_label, title):
    ax.axvspan(nh_start_modis - 0.5, years_full[-1] + 0.5,
               color="gray", alpha=0.1, zorder=0,
               label=f"MODIS period ({nh_start_modis}–{years_full[-1]})")
    ax.plot(years_full, ts_full, "o-", color=color, linewidth=1.5,
            markersize=4, label=series_label)

    tl_full = _trend_line(mk_full, len(years_full))
    ns_full = "" if mk_full.p < alpha else ", n.s."
    kw_full = dict(linewidth=2.5, alpha=0.75) if mk_full.p < alpha \
        else dict(linewidth=2, alpha=0.4)
    ax.plot(years_full, tl_full, "-", color=color,
            label=(f"Full-record: {mk_full.slope:+.4f} M km² yr$^{{-1}}$ "
                   f"(p = {mk_full.p:.3f}{ns_full})"),
            **kw_full)

    tl_ov = _trend_line(mk_ov, len(years_ov))
    ns_ov = "" if mk_ov.p < alpha else ", n.s."
    kw_ov = dict(linewidth=2.5, alpha=0.8) if mk_ov.p < alpha \
        else dict(linewidth=2, alpha=0.45)
    ax.plot(years_ov, tl_ov, "--", color="firebrick",
            label=(f"MODIS period: {mk_ov.slope:+.4f} M km² yr$^{{-1}}$ "
                   f"(p = {mk_ov.p:.3f}{ns_ov})"),
            **kw_ov)

    ax.set_ylabel("Snow extent (M km²)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(years_full[0] - 0.5, years_full[-1] + 0.5)


fig2, (ax_nh, ax_am, ax_eu) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

years_nh_full = nh_full.index.values
years_nh_ov   = nh_overlap.index.values

_plot_extent_panel(
    ax_nh,
    years_nh_full, nh_full.values, mk_nh_full,
    years_nh_ov,   nh_overlap.values, mk_nh_overlap,
    color="steelblue",
    series_label="NH snow extent (all land)",
    title="Full Northern Hemisphere",
)

colors = ["seagreen", "darkorange"]
axes   = [ax_am, ax_eu]
for ax, sd, color in zip(axes, sub_results, colors):
    sd_years_full = sd["full"].index.values
    sd_years_ov   = sd["overlap"].index.values
    _plot_extent_panel(
        ax,
        sd_years_full, sd["full"].values,    sd["mk_full"],
        sd_years_ov,   sd["overlap"].values, sd["mk_overlap"],
        color=color,
        series_label=f"{sd['name']} snow extent",
        title=f"{sd['name']} (lon {sd['lon_min']}° to {sd['lon_max']}°)",
    )

ax_eu.set_xlabel("Year")
fig2.suptitle("NOAA NH Snow Cover Extent — Annual Mean (1967–2025)", fontsize=13, y=1.01)
fig2.tight_layout()
out_png2 = fig_dir.parent / "noaa_nh_extent_trends.png"
fig2.savefig(out_png2, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_png2}")

# =============================================================
# 9. LaTeX table — extent trends
# =============================================================
tex_nh_path = latex_dir / "noaa_nh_extent_trend_summary.tex"
with open(tex_nh_path, "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Mann-Kendall trend analysis of NOAA NH annual mean snow cover "
            "extent (land pixels only) for the full Northern Hemisphere and two "
            "longitudinal subdomains.}\n")
    f.write("\\label{tab:noaa_nh_extent_trend}\n")
    f.write("\\begin{tabular}{llrrr}\n")
    f.write("\\hline\n")
    f.write("Domain & Period & $n$ & Sen's slope (M\\,km$^2$\\,yr$^{-1}$) & $p$-value \\\\\n")
    f.write("\\hline\n")
    rows_ext = [
        ("Full NH",    "1967--2025",              len(nh_full),    mk_nh_full),
        ("Full NH",    f"{nh_start_modis}--2025", len(nh_overlap), mk_nh_overlap),
    ]
    for sd in sub_results:
        rows_ext += [
            (sd["name"], "1967--2025",              len(sd["full"]),    sd["mk_full"]),
            (sd["name"], f"{nh_start_modis}--2025", len(sd["overlap"]), sd["mk_overlap"]),
        ]
    for domain, period, n, result in rows_ext:
        ns = "" if result.p < alpha else " (n.s.)"
        f.write(f"{domain} & {period} & {n} & "
                f"{result.slope:+.4f}{ns} & {result.p:.4f} \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")
print(f"Saved LaTeX table: {tex_nh_path}")
