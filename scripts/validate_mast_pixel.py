"""
Validation: MODIS mast pixel vs in situ observations — Zackenberg
=================================================================
Loads in situ observations from the GEM climate mast and prepares
them for comparison with the MODIS mast-pixel time series.

Variables loaded:
  - Air temperature at 200 cm (hourly)
  - Snow depth (3-hourly)
  - Shortwave incoming radiation (5-min)
  - Shortwave outgoing radiation (5-min)

From shortwave incoming and outgoing, daily albedo is derived as:
  albedo = SRO_daily / SRI_daily  (only during daylight hours)

All variables are resampled to daily values before comparison.

Usage:
    python scripts/validate_mast_pixel.py --site zackenberg
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True, help="Site name matching a config/<site>.yml")
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site = cfg["site"]
gem_dir = Path(cfg["gem_dir"])

# ============================================================
# Helper
# ============================================================

def load_gem_file(path, value_col):
    """Load a GEM tab-separated data file into a datetime-indexed Series."""
    df = pd.read_csv(path, sep="\t", na_values=-9999, encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.set_index("datetime").drop(columns=["Date", "Time", "Quality Flag"])
    return df[value_col].rename(value_col)

# ============================================================
# 1. Air temperature (hourly → daily mean)
# ============================================================
print("Loading air temperature...")
at_path = next(gem_dir.glob("Air_temperature_200cm_60_30min*_data.txt"))
at = load_gem_file(at_path, "AT (°C)")
at_daily = at.resample("1D").mean()
print(f"  {at_path.name}")
print(f"  Range: {at_daily.first_valid_index().date()} to {at_daily.last_valid_index().date()}")
print(f"  Valid days: {at_daily.notna().sum()}")

# ============================================================
# 2. Snow depth (3-hourly → daily mean)
# ============================================================
print("\nLoading snow depth...")
sd_path = next(gem_dir.glob("Snow depth - 180min*_data.txt"))
sd = load_gem_file(sd_path, "SD (m)")
sd_daily = sd.resample("1D").mean()
print(f"  {sd_path.name}")
print(f"  Range: {sd_daily.first_valid_index().date()} to {sd_daily.last_valid_index().date()}")
print(f"  Valid days: {sd_daily.notna().sum()}")

# ============================================================
# 3. Shortwave radiation (5-min → daily sums for albedo)
# ============================================================
print("\nLoading shortwave incoming radiation...")
sri_path = next(gem_dir.glob("Short wave incoming*_data.txt"))
sri = load_gem_file(sri_path, "SRI (W/m2)")
print(f"  {sri_path.name}")

print("Loading shortwave outgoing radiation...")
sro_path = next(gem_dir.glob("Short wave outgoing*_data.txt"))
sro = load_gem_file(sro_path, "SRO (W/m2)")
print(f"  {sro_path.name}")

# Daily albedo from daily sums — more robust than mean of instantaneous values.
# Mask days where total incoming is too low (polar night / overcast) to avoid
# division noise; threshold of 1 W/m2 mean (~288 W/m2 daily sum at 5-min res).
sri_daily = sri.resample("1D").mean()
sro_daily = sro.resample("1D").mean()

sri_daily_sum = sri.resample("1D").sum()
sro_daily_sum = sro.resample("1D").sum()

albedo_daily_raw = (sro_daily_sum / sri_daily_sum).where(sri_daily > 1).clip(0, 1)

# Fill gaps within the daylight season by linear interpolation.
# Polar night days (sri_daily <= 1) remain NaN — only gaps between valid
# daylight observations are filled. A 14-day limit avoids bridging over
# extended instrument outages.
daylight = sri_daily > 1
albedo_daily = albedo_daily_raw.copy()
albedo_daily[daylight] = (
    albedo_daily_raw[daylight]
    .interpolate(method="time", limit=14, limit_area="inside")
)

n_filled = int(albedo_daily.notna().sum() - albedo_daily_raw.notna().sum())
print(f"  Radiation range: {sri_daily.first_valid_index().date()} "
      f"to {sri_daily.last_valid_index().date()}")
print(f"  Valid albedo days (raw):    {albedo_daily_raw.notna().sum()}")
print(f"  Valid albedo days (filled): {albedo_daily.notna().sum()} "
      f"({n_filled} gaps filled by interpolation)")

# ============================================================
# 4. Combine into a single daily dataframe
# ============================================================
print("\nCombining into daily dataframe...")
insitu = pd.DataFrame({
    "air_temp_c":     at_daily,
    "snow_depth_m":   sd_daily,
    "sri_wm2":        sri_daily,
    "sro_wm2":        sro_daily,
    "albedo":         albedo_daily,
    "albedo_raw":     albedo_daily_raw,
})

print(f"  Combined shape: {insitu.shape}")
print(f"  Date range: {insitu.index[0].date()} to {insitu.index[-1].date()}")
print(f"\n{insitu.describe().round(3).to_string()}")

# ============================================================
# 5. Albedo threshold exploration
# ============================================================
print("\n" + "=" * 60)
print("Albedo by month (to guide snow-free threshold selection)")
print("=" * 60)

albedo_valid = insitu["albedo"].dropna()
monthly = albedo_valid.groupby(albedo_valid.index.month)

print(f"\n{'Month':>10} {'Mean':>8} {'Std':>8} {'P10':>8} {'P25':>8} {'P75':>8} {'P90':>8} {'N':>6}")
print("-" * 66)
month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
               7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
for month, vals in monthly:
    print(f"{month_names[month]:>10} {vals.mean():8.3f} {vals.std():8.3f} "
          f"{vals.quantile(0.10):8.3f} {vals.quantile(0.25):8.3f} "
          f"{vals.quantile(0.75):8.3f} {vals.quantile(0.90):8.3f} {len(vals):6d}")

july  = albedo_valid[albedo_valid.index.month == 7]
march = albedo_valid[albedo_valid.index.month == 3]
print(f"\nKey comparison:")
print(f"  July  (snow-free):    mean={july.mean():.3f},  std={july.std():.3f},  "
      f"P10={july.quantile(0.10):.3f},  P90={july.quantile(0.90):.3f}")
print(f"  March (snow-covered): mean={march.mean():.3f},  std={march.std():.3f},  "
      f"P10={march.quantile(0.10):.3f},  P90={march.quantile(0.90):.3f}")
midpoint = (july.mean() + march.mean()) / 2
print(f"\n  Midpoint between means: {midpoint:.3f}")

# ============================================================
# 6. Albedo threshold vs snow depth
# ============================================================
print("\n" + "=" * 60)
print("Albedo threshold alignment with snow depth")
print("=" * 60)

# Work on days where both albedo and snow depth are available
both = insitu[["albedo", "snow_depth_m"]].dropna()
snow_depth_threshold = 0.02  # m — treat <= this as snow-free to allow for sensor noise

sd_snow_free = (both["snow_depth_m"] <= snow_depth_threshold).astype(int)

thresholds = np.arange(0.2, 0.85, 0.05)
print(f"\n  Snow depth reference: snow-free if SD <= {snow_depth_threshold} m")
print(f"  Overlapping days with both variables: {len(both)}\n")
print(f"  {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
print("  " + "-" * 52)

best_f1, best_thresh = 0, None
for thresh in thresholds:
    albedo_snow_free = (both["albedo"] < thresh).astype(int)
    tp = ((albedo_snow_free == 1) & (sd_snow_free == 1)).sum()
    fp = ((albedo_snow_free == 1) & (sd_snow_free == 0)).sum()
    tn = ((albedo_snow_free == 0) & (sd_snow_free == 0)).sum()
    fn = ((albedo_snow_free == 0) & (sd_snow_free == 1)).sum()
    accuracy  = (tp + tn) / len(both)
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall    = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
    marker = " <--" if f1 == max(f1, best_f1) and not np.isnan(f1) else ""
    if not np.isnan(f1) and f1 > best_f1:
        best_f1, best_thresh = f1, thresh
    print(f"  {thresh:>10.2f} {accuracy:>10.3f} {precision:>10.3f} {recall:>10.3f} {f1:>8.3f}{marker}")

print(f"\n  Best threshold by F1: {best_thresh:.2f} (F1={best_f1:.3f})")

# --- Plot 1: albedo vs snow depth scatter ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(both["snow_depth_m"], both["albedo"], alpha=0.2, s=5, color="steelblue")
ax.axhline(best_thresh, color="firebrick", linewidth=1.5, linestyle="--",
           label=f"Best threshold ({best_thresh:.2f})")
ax.axhline(midpoint, color="darkorange", linewidth=1.5, linestyle=":",
           label=f"Midpoint ({midpoint:.2f})")
ax.axvline(snow_depth_threshold, color="gray", linewidth=1, linestyle="--",
           label=f"SD reference ({snow_depth_threshold} m)")
ax.set_xlabel("Snow depth (m)")
ax.set_ylabel("Daily albedo")
ax.set_title("Albedo vs snow depth")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Plot 2: threshold performance curve ---
ax2 = axes[1]
accs, precs, recs, f1s = [], [], [], []
thresh_range = np.arange(0.1, 0.95, 0.01)
for thresh in thresh_range:
    albedo_snow_free = (both["albedo"] < thresh).astype(int)
    tp = ((albedo_snow_free == 1) & (sd_snow_free == 1)).sum()
    fp = ((albedo_snow_free == 1) & (sd_snow_free == 0)).sum()
    tn = ((albedo_snow_free == 0) & (sd_snow_free == 0)).sum()
    fn = ((albedo_snow_free == 0) & (sd_snow_free == 1)).sum()
    accs.append((tp + tn) / len(both))
    p = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    r = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    precs.append(p)
    recs.append(r)
    f1s.append(2 * p * r / (p + r) if (p + r) > 0 else np.nan)

ax2.plot(thresh_range, accs,  label="Accuracy",  color="steelblue")
ax2.plot(thresh_range, precs, label="Precision", color="darkorange")
ax2.plot(thresh_range, recs,  label="Recall",    color="teal")
ax2.plot(thresh_range, f1s,   label="F1",        color="firebrick", linewidth=2)
ax2.axvline(best_thresh, color="firebrick", linewidth=1.5, linestyle="--",
            label=f"Best F1 ({best_thresh:.2f})")
ax2.set_xlabel("Albedo threshold (snow-free if albedo < threshold)")
ax2.set_ylabel("Score")
ax2.set_title("Threshold performance vs snow depth")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.1, 0.95)
ax2.set_ylim(0, 1.05)

fig.suptitle(f"{site.capitalize()}: Albedo threshold alignment with snow depth")
fig.tight_layout()

out_dir = Path(f"figures/{site}/")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "albedo_threshold_vs_snow_depth.png", dpi=150)
print(f"\n  Saved: figures/{site}/albedo_threshold_vs_snow_depth.png")

# ============================================================
# 7. Annual snow-free days: albedo vs MODIS mast pixel
# ============================================================
print("\n" + "=" * 60)
print("Annual snow-free days: in situ albedo vs MODIS mast pixel")
print("=" * 60)

albedo_threshold = 0.4
modis_thresholds = [10, 20, 30, 40, 50, 60, 70]

# --- In situ: snow-free days per year from albedo ---
# Albedo is only measurable during daylight — at 74°N polar night covers
# ~100 days, so require 150 valid days rather than a full-year threshold
albedo_col = insitu["albedo"]
sfd_insitu_rows = []
for year, grp in albedo_col.groupby(albedo_col.index.year):
    n_valid = grp.notna().sum()
    if n_valid < 150:
        print(f"  In situ {year}: skipped ({n_valid} valid albedo days)")
        continue
    sfd_insitu_rows.append({"year": year, "sfd_insitu": int((grp < albedo_threshold).sum())})

sfd_insitu = pd.DataFrame(sfd_insitu_rows).set_index("year")

# --- MODIS: load mast-pixel CSV, compute SFD for all thresholds ---
modis_csv = Path(f"results/csvs/{site}/mast_pixel_snow_{site}.csv")
if not modis_csv.exists():
    raise FileNotFoundError(f"MODIS mast pixel CSV not found: {modis_csv}\n"
                            "Run get_snow_free_days.py first.")

modis_px = pd.read_csv(modis_csv, parse_dates=["date"])
modis_px["year"] = modis_px["date"].dt.year

sfd_modis = sfd_insitu[[]].copy()  # empty frame with same index, filled below
for thresh in modis_thresholds:
    col = f"snow_lt{thresh}"
    rows = []
    for year, grp in modis_px.groupby("year"):
        if grp[col].notna().sum() < 150:
            continue
        rows.append({"year": year, col: int((grp[col] == 1).sum())})
    sfd_modis = sfd_modis.join(
        pd.DataFrame(rows).set_index("year"), how="outer"
    )

# --- Load area-mean MODIS snow-free days for all thresholds ---
area_csv = Path(f"results/csvs/{site}/snow_free_days.csv")
if not area_csv.exists():
    raise FileNotFoundError(f"Area snow-free days CSV not found: {area_csv}\n"
                            "Run get_snow_free_days.py first.")
sfd_area = pd.read_csv(area_csv).set_index("year")
sfd_area = sfd_area[[f"sfd_lt{t}" for t in modis_thresholds]]

# --- Print summary for primary threshold ---
primary = 40
comp_primary = (sfd_insitu
                .join(sfd_modis[[f"snow_lt{primary}"]], how="inner")
                .join(sfd_area[[f"sfd_lt{primary}"]], how="left")
                .dropna(subset=[f"snow_lt{primary}"]))

print(f"\n  Albedo threshold: < {albedo_threshold}")
print(f"\n  {'Year':>6} {'In situ':>10} {'Mast px (40%)':>14} {'Area (40%)':>11}")
print("  " + "-" * 45)
for year, row in comp_primary.iterrows():
    print(f"  {year:>6} {row['sfd_insitu']:>10.0f} {row[f'snow_lt{primary}']:>14.0f} "
          f"{row[f'sfd_lt{primary}']:>11.1f}")

print(f"\n  Threshold  Correlation  Bias (d)")
print("  " + "-" * 32)
for thresh in modis_thresholds:
    col = f"snow_lt{thresh}"
    merged = sfd_insitu.join(sfd_modis[[col]], how="inner").dropna()
    if len(merged) < 3:
        continue
    r    = merged["sfd_insitu"].corr(merged[col])
    bias = (merged[col] - merged["sfd_insitu"]).mean()
    print(f"  {thresh:>9}%  {r:>11.3f}  {bias:>+8.1f}")

# --- Plot ---
cmap   = plt.cm.viridis
colors = cmap(np.linspace(0.1, 0.9, len(modis_thresholds)))

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: time series — in situ + mast pixel all thresholds
ax = axes2[0]
ax.plot(sfd_insitu.index, sfd_insitu["sfd_insitu"], "o-", color="steelblue",
        linewidth=2, markersize=5, label=f"In situ (albedo < {albedo_threshold})", zorder=5)
for thresh, color in zip(modis_thresholds, colors):
    col = f"snow_lt{thresh}"
    valid = sfd_modis[col].dropna()
    ax.plot(valid.index, valid.values, "--", color=color,
            linewidth=1, markersize=3, marker="s", label=f"Mast pixel NDSI < {thresh}%")
ax.set_xlabel("Year")
ax.set_ylabel("Snow-free days per year")
ax.set_title(f"{site.capitalize()}: In situ vs MODIS mast pixel")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

def scatter_panel(ax, x_vals, y_vals, years, xlabel, ylabel, title):
    lims = [0, max(x_vals.max(), y_vals.max()) + 10]
    ax.scatter(x_vals, y_vals, color="steelblue", s=40, zorder=3)
    ax.plot(lims, lims, "--", color="gray", linewidth=1, label="1:1 line")
    for year, xv, yv in zip(years, x_vals, y_vals):
        ax.annotate(str(year), (xv, yv),
                    fontsize=7, textcoords="offset points", xytext=(4, 2))
    r    = np.corrcoef(x_vals, y_vals)[0, 1]
    bias = (y_vals - x_vals).mean()
    ax.text(0.05, 0.95, f"r = {r:.3f}\nBias = {bias:+.1f} d",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel 2: scatter — in situ vs mast pixel, primary threshold only
ax2 = axes2[1]
col = f"snow_lt{primary}"
merged2 = sfd_insitu.join(sfd_modis[[col]], how="inner").dropna()
scatter_panel(ax2,
              x_vals=merged2["sfd_insitu"].values,
              y_vals=merged2[col].values,
              years=merged2.index,
              xlabel=f"In situ (albedo < {albedo_threshold})",
              ylabel=f"MODIS mast pixel (NDSI < {primary}%)",
              title=f"{site.capitalize()}: In situ vs mast pixel")

# Panel 3: scatter — mast pixel vs area mean, primary threshold only
ax3 = axes2[2]
area_col = f"sfd_lt{primary}"
merged3 = sfd_modis[[col]].join(sfd_area[[area_col]], how="inner").dropna()
scatter_panel(ax3,
              x_vals=merged3[col].values,
              y_vals=merged3[area_col].values,
              years=merged3.index,
              xlabel=f"MODIS mast pixel (NDSI < {primary}%)",
              ylabel=f"MODIS area mean (NDSI < {primary}%)",
              title=f"{site.capitalize()}: Mast pixel vs area mean")

fig2.tight_layout()
fig2.savefig(out_dir / "sfd_insitu_vs_modis.png", dpi=150)
print(f"\n  Saved: figures/{site}/sfd_insitu_vs_modis.png")

# ============================================================
# 8. Per-year snow depth and albedo plots
# ============================================================
print("\nGenerating per-year snow depth and albedo plots...")

insitu_dir = out_dir / "in_situ"
insitu_dir.mkdir(parents=True, exist_ok=True)

years_with_data = insitu[["snow_depth_m", "albedo"]].dropna(how="all").index.year.unique()

for year in sorted(years_with_data):
    yearly = insitu.loc[str(year)]
    if yearly["albedo"].notna().sum() == 0 and yearly["snow_depth_m"].notna().sum() == 0:
        continue

    fig3, ax_sd = plt.subplots(figsize=(12, 4))
    ax_al = ax_sd.twinx()

    doy = yearly.index.dayofyear

    ax_sd.plot(doy, yearly["snow_depth_m"], color="steelblue", linewidth=1.2,
               label="Snow depth (m)")
    ax_al.plot(doy, yearly["albedo"], color="darkorange", linewidth=1.2,
               linestyle="--", label="Albedo (gap-filled)")
    ax_al.plot(doy, yearly["albedo_raw"], color="darkorange", linewidth=0,
               marker=".", markersize=3, label="Albedo (observed)")
    ax_al.axhline(albedo_threshold, color="darkorange", linewidth=0.8,
                  linestyle=":", alpha=0.6, label=f"Albedo threshold ({albedo_threshold})")

    ax_sd.set_xlabel("Day of year")
    ax_sd.set_ylabel("Snow depth (m)", color="steelblue")
    ax_al.set_ylabel("Albedo", color="darkorange")
    ax_sd.tick_params(axis="y", labelcolor="steelblue")
    ax_al.tick_params(axis="y", labelcolor="darkorange")
    ax_sd.set_xlim(1, 366)
    ax_sd.set_ylim(bottom=0)
    ax_al.set_ylim(0, 1.05)
    ax_sd.grid(True, alpha=0.3)

    lines_sd, labels_sd = ax_sd.get_legend_handles_labels()
    lines_al, labels_al = ax_al.get_legend_handles_labels()
    ax_sd.legend(lines_sd + lines_al, labels_sd + labels_al, loc="upper right", fontsize=8)

    fig3.suptitle(f"{site.capitalize()} {year}: snow depth and albedo", fontsize=11)
    fig3.tight_layout()
    fig3.savefig(insitu_dir / f"snow_depth_albedo_{year}.png", dpi=150)
    plt.close(fig3)

print(f"  Saved {len(list(insitu_dir.glob('*.png')))} plots to figures/{site}/in_situ/")
