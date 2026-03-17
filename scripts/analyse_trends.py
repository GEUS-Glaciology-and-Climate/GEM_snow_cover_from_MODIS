"""
Trend Analysis of Snow-Free Days — Zackenberg
==============================================
Reads snow-free day time series from CSV (produced by
analyse_snow_free_days.py) and tests for significant trends in:
  1. Mean snow-free days per year (threshold < 40%)
  2. Interannual variability: 5-year rolling std dev

Uses Mann-Kendall test and Sen's slope estimator.
Only displays trend lines if statistically significant.

Usage:
    python analyse_trends.py --site zackenberg
"""
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymannkendall as mk
from pathlib import Path

# --- Config ---
threshold = 40
alpha = 0.05
rolling_window = 5

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True, help="Site name matching a config/<site>.yml")
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site              = cfg["site"]
csv_path           = Path(f"./results/csvs/{site}/snow_free_days.csv")
fig_dir           = Path(f"./figures/{site}/")  
fig_dir.mkdir(parents=True, exist_ok=True)

# --- Load data ---
df = pd.read_csv(csv_path)
years = df["year"].values
mean_sfd = df[f"sfd_lt{threshold}"].values

print(f"Loaded {len(years)} years from {csv_path}")
print(f"Year range: {years[0]}–{years[-1]}")



# ============================================================
# 1. Trend in mean snow-free days
# ============================================================
print("\n" + "=" * 60)
print(f"Trend analysis: mean snow-free days (threshold < {threshold}%)")
print("=" * 60)

mk_mean = mk.original_test(mean_sfd, alpha=alpha)
print(f"  Mann-Kendall trend:  {mk_mean.trend}")
print(f"  p-value:             {mk_mean.p:.4f}")
print(f"  Sen's slope:         {mk_mean.slope:.3f} days/year")
print(f"  Intercept:           {mk_mean.intercept:.3f}")
print(f"  Significant (α={alpha}): {mk_mean.p < alpha}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(years, mean_sfd, "o-", color="steelblue", linewidth=1.5,
        markersize=5, label="Mean snow-free days")

if mk_mean.p < alpha:
    trend_line = mk_mean.intercept + mk_mean.slope * np.arange(len(years))
    ax.plot(years, trend_line, "--", color="firebrick", linewidth=2,
            label=f"Sen's slope: {mk_mean.slope:.2f} d/yr (p={mk_mean.p:.3f})")
else:
    ax.text(0.5, 0.95, f"No significant trend (p={mk_mean.p:.2f})",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, color="gray", style="italic")

ax.set_xlabel("Year")
ax.set_ylabel(f"Mean snow-free days (NDSI < {threshold}%)")
ax.set_title(f"{site.capitalize()}: Snow-Free Days per Year (threshold < {threshold}%)")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
fig.tight_layout()
fig.savefig(fig_dir / "trend_mean_snow_free_days.png", dpi=150)
print(f"\n  Saved: trend_mean_snow_free_days.png")

# ============================================================
# 2. Trend in interannual variability (rolling std dev)
# ============================================================
print("\n" + "=" * 60)
print(f"Trend analysis: {rolling_window}-year rolling std dev of mean snow-free days")
print("=" * 60)

sfd_series = pd.Series(mean_sfd, index=years)
rolling_std = sfd_series.rolling(window=rolling_window, center=True, min_periods=3).std()

# Drop NaN edges for trend testing
valid = rolling_std.dropna()
years_var = valid.index.values
vals_var = valid.values

mk_var = mk.original_test(vals_var, alpha=alpha)
print(f"  Mann-Kendall trend:  {mk_var.trend}")
print(f"  p-value:             {mk_var.p:.4f}")
print(f"  Sen's slope:         {mk_var.slope:.3f} days/year")
print(f"  Intercept:           {mk_var.intercept:.3f}")
print(f"  Significant (α={alpha}): {mk_var.p < alpha}")

# --- Plot ---
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(years_var, vals_var, "o-", color="darkorange", linewidth=1.5,
         markersize=5, label=f"{rolling_window}-year rolling std dev")

if mk_var.p < alpha:
    trend_line2 = mk_var.intercept + mk_var.slope * np.arange(len(years_var))
    ax2.plot(years_var, trend_line2, "--", color="firebrick", linewidth=2,
             label=f"Sen's slope: {mk_var.slope:.2f} d/yr (p={mk_var.p:.3f})")
else:
    ax2.text(0.5, 0.95, f"No significant trend (p={mk_var.p:.2f})",
             transform=ax2.transAxes, ha="center", va="top",
             fontsize=11, color="gray", style="italic")

ax2.set_xlabel("Year")
ax2.set_ylabel("Rolling std dev (days)")
ax2.set_title(f"{site.capitalize()}: Interannual Variability of Snow-Free Days "
              f"({rolling_window}-year window, threshold < {threshold}%)")
ax2.legend(loc="best")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(years[0] - 0.5, years[-1] + 0.5)
fig2.tight_layout()
fig2.savefig(fig_dir / "trend_interannual_variability.png", dpi=150)
print(f"\n  Saved: trend_interannual_variability.png")

# ============================================================
# 3. Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"{'Metric':<35} {'Slope':>10} {'p-value':>10} {'Significant':>12}")
print("-" * 67)
print(f"{'Mean snow-free days':<35} {mk_mean.slope:>10.3f} {mk_mean.p:>10.4f} "
      f"{'Yes' if mk_mean.p < alpha else 'No':>12}")
print(f"{'Interannual variability (5yr)':<35} {mk_var.slope:>10.3f} {mk_var.p:>10.4f} "
      f"{'Yes' if mk_var.p < alpha else 'No':>12}")

# LaTeX table
tex_path = fig_dir / "trend_summary.tex"
with open(tex_path, "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Mann-Kendall trend analysis of snow-free days, "
            f"{site.capitalize()} (NDSI threshold $<$ {threshold}\\%).}}\n")
    f.write("\\label{tab:trend_summary}\n")
    f.write("\\begin{tabular}{lrrr}\n")
    f.write("\\hline\n")
    f.write("Metric & Sen's slope (d\\,yr$^{-1}$) & $p$-value & Significant \\\\\n")
    f.write("\\hline\n")
    for name, result in [("Mean snow-free days", mk_mean),
                         (f"Interannual variability ({rolling_window}\\,yr)", mk_var)]:
        sig = "Yes" if result.p < alpha else "No"
        f.write(f"{name} & {result.slope:.3f} & {result.p:.4f} & {sig} \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"\nSaved LaTeX table: {tex_path}")
