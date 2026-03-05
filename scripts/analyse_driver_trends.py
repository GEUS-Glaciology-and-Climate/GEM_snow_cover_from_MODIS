"""
Trend Analysis of Climate Drivers — Zackenberg
===============================================
Tests for trends in the key drivers of snow-free days:
  1. PDD (Positive Degree Days) — calendar year
  2. Winter precipitation — hydrological year (Sept–Sept)
  3. Snow-free days (for comparison)

Uses Mann-Kendall test and Sen's slope estimator.
Produces a combined figure for direct visual comparison.

Usage:
    python analyse_driver_trends.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymannkendall as mk
from pathlib import Path

# --- Config ---
workingdir = Path("/home/shl/mdrev/projects/modis/snow_cover")
csv_dir = workingdir / "results" / "csvs"
fig_dir = workingdir / "figures" / "zackenberg"
fig_dir.mkdir(parents=True, exist_ok=True)

threshold = 40
alpha = 0.05

# --- Load data ---
print("Loading data...")
sfd = pd.read_csv(csv_dir / "snow_free_days.csv")
climate = pd.read_csv(csv_dir / "climate_predictors_zackenberg.csv")

df = pd.merge(sfd[["year", f"sfd_lt{threshold}"]], climate, on="year", how="inner")
df = df.dropna()
df = df.rename(columns={
    f"sfd_lt{threshold}": "snow_free_days",
    "pdd_carra": "PDD",
    "winter_precip_mm": "winter_precip",
})

print(f"Years: {df['year'].min()}–{df['year'].max()} (n={len(df)})")

# --- Define variables to test ---
variables = {
    "Snow-free days": {
        "data": df["snow_free_days"].values,
        "unit": "days",
        "color": "steelblue",
        "slope_unit": "days/yr",
    },
    "PDD": {
        "data": df["PDD"].values,
        "unit": "°C·days",
        "color": "firebrick",
        "slope_unit": "°C·d/yr",
    },
    "Winter precipitation": {
        "data": df["winter_precip"].values,
        "unit": "mm",
        "color": "teal",
        "slope_unit": "mm/yr",
    },
}

years = df["year"].values

# ============================================================
# 1. Mann-Kendall tests
# ============================================================
print("\n" + "=" * 60)
print("Mann-Kendall trend tests")
print("=" * 60)
print(f"{'Variable':<25} {'Slope':>10} {'p-value':>10} {'Significant':>12}")
print("-" * 57)

mk_results = {}
for name, var in variables.items():
    result = mk.original_test(var["data"], alpha=alpha)
    mk_results[name] = result
    sig = "Yes" if result.p < alpha else "No"
    print(f"{name:<25} {result.slope:>10.3f} {result.p:>10.4f} {sig:>12}")

# ============================================================
# 2. Combined three-panel figure
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

for ax, (name, var) in zip(axes, variables.items()):
    result = mk_results[name]
    data = var["data"]

    ax.plot(years, data, "o-", color=var["color"], linewidth=1.5,
            markersize=5, label=name)

    if result.p < alpha:
        trend_line = result.intercept + result.slope * np.arange(len(years))
        ax.plot(years, trend_line, "--", color="black", linewidth=2,
                label=f"Sen's slope: {result.slope:.2f} {var['slope_unit']} "
                      f"(p={result.p:.3f})")
    else:
        ax.text(0.98, 0.95, f"No significant trend (p={result.p:.2f})",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color="gray", style="italic")

    ax.set_ylabel(f"{name} ({var['unit']})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Year")
axes[0].set_title("Zackenberg: Trends in Snow-Free Days and Climate Drivers")
fig.tight_layout()
fig.savefig(fig_dir / "driver_trends_combined.png", dpi=150)
print(f"\nSaved: driver_trends_combined.png")

# ============================================================
# 3. Summary narrative
# ============================================================
print("\n" + "=" * 60)
print("Interpretation")
print("=" * 60)

sfd_sig = mk_results["Snow-free days"].p < alpha
pdd_sig = mk_results["PDD"].p < alpha
wp_sig = mk_results["Winter precipitation"].p < alpha

if not sfd_sig and not pdd_sig and not wp_sig:
    print("  No significant trends in snow-free days or either driver.")
    print("  The absence of a trend in snow-free days is consistent with")
    print("  the absence of trends in the underlying climate drivers.")
elif not sfd_sig and (pdd_sig or wp_sig):
    print("  No significant trend in snow-free days, but trends detected")
    print("  in one or more drivers. This may indicate compensating effects")
    print("  between drivers, or that the trending driver has a weaker")
    print("  influence on snow-free days.")
    if pdd_sig and not wp_sig:
        print("  Specifically: PDD is trending but winter precipitation is not.")
    elif wp_sig and not pdd_sig:
        print("  Specifically: winter precipitation is trending but PDD is not.")
    else:
        print("  Both drivers show significant trends.")
elif sfd_sig:
    trending_drivers = []
    if pdd_sig:
        trending_drivers.append("PDD")
    if wp_sig:
        trending_drivers.append("winter precipitation")
    if trending_drivers:
        print(f"  Snow-free days and {', '.join(trending_drivers)} all show")
        print("  significant trends, consistent with a climate-driven change.")
    else:
        print("  Snow-free days show a significant trend but neither driver does.")
        print("  This suggests other factors may be influencing snow cover duration.")

# ============================================================
# 4. LaTeX table
# ============================================================
tex_path = fig_dir / "driver_trends.tex"
with open(tex_path, "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Mann-Kendall trend analysis of snow-free days and "
            "climate drivers, Zackenberg.}\n")
    f.write("\\label{tab:driver_trends}\n")
    f.write("\\begin{tabular}{lrrr}\n")
    f.write("\\hline\n")
    f.write("Variable & Sen's slope & $p$-value & Significant \\\\\n")
    f.write("\\hline\n")
    for name, var in variables.items():
        result = mk_results[name]
        sig = "Yes" if result.p < alpha else "No"
        unit = var["slope_unit"]
        f.write(f"{name} ({unit}) & {result.slope:.3f} & "
                f"{result.p:.4f} & {sig} \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"\nSaved LaTeX table: {tex_path}")
print("\nDone.")
