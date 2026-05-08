"""
Validate CARRA winter precipitation against in situ spring snow depth
=====================================================================
Compares the CARRA-derived cold-season precipitation (hydrological year,
T < 0°C) with the maximum in situ snow depth observed each spring
(March–May), which represents the end-of-winter snowpack.

A positive relationship between the two is expected: heavier winters
should produce deeper spring snowpacks.

Usage:
    python scripts/validate_winter_precip.py --site zackenberg
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

site        = cfg["site"]
gem_dir     = Path(cfg["gem_dir"])
sd_file     = cfg["insitu_snowdepth"]
col_sd      = cfg.get("insitu_col_snowdepth",      "SD (m)")
time_col_sd = cfg.get("insitu_time_col_snowdepth", "Time")

csv_dir = Path(f"results/csvs/{site}/")
fig_dir = Path(f"figures/{site}/")
csv_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

SPRING_MONTHS = (1,2,3, 4, 5)   # March–May: end-of-winter snowpack peak

# ============================================================
# 1. Load winter precipitation (CARRA, already derived)
# ============================================================
pred_path = csv_dir / f"climate_predictors_{site}.csv"
if not pred_path.exists():
    raise FileNotFoundError(
        f"Climate predictors not found: {pred_path}\n"
        "Run derive_climate_predictors.py first."
    )

predictors = pd.read_csv(pred_path, index_col="year")
winter_precip = predictors["winter_precip_mm"].dropna()
print(f"Loaded winter precip: {winter_precip.index[0]}–{winter_precip.index[-1]} "
      f"({len(winter_precip)} years)")

# ============================================================
# 2. Load in situ snow depth and compute spring maximum
# ============================================================
print("Loading in situ snow depth...")
sd_path = next(gem_dir.glob(sd_file))
df = pd.read_csv(sd_path, sep="\t", na_values=-9999, encoding="utf-8-sig")
df["datetime"] = pd.to_datetime(df["Date"] + " " + df[time_col_sd])
df = df.set_index("datetime")
sd = df[col_sd]

sd_daily = sd.resample("1D").mean()

# Spring maximum: max daily mean in March–May
spring = sd_daily[sd_daily.index.month.isin(SPRING_MONTHS)]

spring_max_rows = []
for year, grp in spring.groupby(spring.index.year):
    n_valid = grp.notna().sum()
    if n_valid < 25:   # require at least 30 valid spring days
        print(f"  {year}: skipped ({n_valid} valid spring days)")
        continue
    spring_max_rows.append({"year": year, "spring_max_sd_m": grp.max()})

spring_max = pd.DataFrame(spring_max_rows).set_index("year")
print(f"Spring max snow depth: {spring_max.index[0]}–{spring_max.index[-1]} "
      f"({len(spring_max)} years)")

# ============================================================
# 3. Merge and summarise
# ============================================================
df_cmp = winter_precip.rename("winter_precip_mm").to_frame().join(
    spring_max, how="inner"
).dropna()

print(f"\nOverlapping years: {df_cmp.index[0]}–{df_cmp.index[-1]} (n={len(df_cmp)})")
print(f"\n{'Year':>6}  {'Winter precip (mm)':>18}  {'Spring max SD (m)':>17}")
print("  " + "-" * 42)
for year, row in df_cmp.iterrows():
    print(f"  {year:>6}  {row['winter_precip_mm']:>18.1f}  {row['spring_max_sd_m']:>17.3f}")

r       = df_cmp["winter_precip_mm"].corr(df_cmp["spring_max_sd_m"])
r2      = r ** 2
bias_mm = df_cmp["winter_precip_mm"].mean()   # just for context
print(f"\n  Pearson r  = {r:.3f}")
print(f"  R²         = {r2:.3f}")

# Save merged CSV
out_csv = csv_dir / f"winter_precip_vs_spring_snowdepth_{site}.csv"
df_cmp.to_csv(out_csv)
print(f"\nSaved: {out_csv}")

# ============================================================
# 4. Figures
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.subplots_adjust(wspace=0.35, top=0.90, bottom=0.13, left=0.09, right=0.97)

x = df_cmp["winter_precip_mm"].values
y = df_cmp["spring_max_sd_m"].values
years = df_cmp.index.values

# --- Panel (a): scatter ---
ax = axes[0]
ax.scatter(x, y, color="steelblue", s=40, zorder=3)
for yr, xv, yv in zip(years, x, y):
    ax.annotate(str(yr), (xv, yv),
                fontsize=6, textcoords="offset points", xytext=(3, 2))

coeffs  = np.polyfit(x, y, 1)
fit_x   = np.linspace(x.min() * 0.9, x.max() * 1.1, 100)
ax.plot(fit_x, np.polyval(coeffs, fit_x), "-", color="firebrick",
        linewidth=1.5, label=f"Fit: y = {coeffs[0]:.4f}x + {coeffs[1]:.2f}")
ax.text(0.05, 0.95, f"r = {r:.3f}\nR² = {r2:.3f}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
ax.set_xlabel("CARRA winter precipitation (mm)", fontsize=9)
ax.set_ylabel("In situ spring max snow depth (m)", fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=8)
ax.text(-0.13, 1.02, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold")

# --- Panel (b): time series ---
ax2 = axes[1]
ax2_r = ax2.twinx()

ax2.plot(df_cmp.index, df_cmp["winter_precip_mm"], "o-",
         color="steelblue", linewidth=1.5, markersize=4,
         label="Winter precip (mm)")
ax2_r.plot(df_cmp.index, df_cmp["spring_max_sd_m"], "s--",
           color="firebrick", linewidth=1.5, markersize=4,
           label="Spring max SD (m)")

ax2.set_xlabel("Year", fontsize=9)
ax2.set_ylabel("Winter precipitation (mm)", color="steelblue", fontsize=9)
ax2_r.set_ylabel("Spring max snow depth (m)", color="firebrick", fontsize=9)
ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=8)
ax2_r.tick_params(axis="y", labelcolor="firebrick", labelsize=8)
ax2.tick_params(axis="x", labelsize=8)
ax2.grid(True, alpha=0.3)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
ax2.text(-0.13, 1.02, "(b)", transform=ax2.transAxes, fontsize=10, fontweight="bold")

fig.suptitle(
    f"{site.capitalize()}: CARRA winter precip vs in situ spring max snow depth "
    f"(Mar–May)",
    fontsize=9,
)

out_fig = fig_dir / "winter_precip_vs_spring_snowdepth.png"
fig.savefig(out_fig, dpi=300)
print(f"Saved: {out_fig}")

out_pdf = fig_dir / "winter_precip_vs_spring_snowdepth.pdf"
fig.savefig(out_pdf)
print(f"Saved: {out_pdf}")
