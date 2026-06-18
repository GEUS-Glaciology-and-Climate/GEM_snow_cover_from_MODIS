"""
Trend Analysis of Climate Drivers
===============================================
Tests for trends in the key drivers of snow-free days:
  1. PDD (Positive Degree Days) — calendar year
  2. Winter precipitation — hydrological year (Oct–Sep)
  3. Snow-free days (for comparison)

Uses Mann-Kendall test and Sen's slope estimator.
Produces a combined figure for direct visual comparison.

Usage:
    python analyse_driver_trends.py --site zackenberg
"""
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymannkendall as mk
from pathlib import Path

# --- Config ---
workingdir = Path("/home/shl/mdrev/projects/modis/snow_cover")

threshold = 40
alpha = 0.05

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True, help="Site name matching a config/<site>.yml")
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site              = cfg["site"]
csv_dir           = Path(f"{workingdir}/results/csvs/{site}/")
fig_dir           = Path(f"{workingdir}/figures/{site}/")
fig_dir.mkdir(parents=True, exist_ok=True)
latex_dir         = Path(f"{workingdir}/results/latex/{site}/")
latex_dir.mkdir(parents=True, exist_ok=True)



# --- Load data ---
print("Loading data...")
sfd = pd.read_csv(f"{csv_dir}/snow_free_days.csv")
climate = pd.read_csv(f"{csv_dir}/climate_predictors_{site}.csv")

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

# --- Sub-period 2000–2025 ---
df_sub = df[(df["year"] >= 2000) & (df["year"] <= 2025)].copy()
years_sub = df_sub["year"].values
print(f"Sub-period: {years_sub.min()}–{years_sub.max()} (n={len(df_sub)})")

variables_sub = {
    "Snow-free days": {**variables["Snow-free days"], "data": df_sub["snow_free_days"].values},
    "PDD":            {**variables["PDD"],            "data": df_sub["PDD"].values},
    "Winter precipitation": {**variables["Winter precipitation"], "data": df_sub["winter_precip"].values},
}

# ============================================================
# 1. Mann-Kendall tests — full period
# ============================================================
def run_mk(vars_dict, alpha):
    results = {}
    for name, var in vars_dict.items():
        result = mk.original_test(var["data"], alpha=alpha)
        results[name] = result
    return results

def print_mk_table(mk_res, variables, label):
    print(f"\n{'=' * 60}")
    print(f"Mann-Kendall trend tests — {label}")
    print("=" * 60)
    print(f"{'Variable':<25} {'Slope':>10} {'p-value':>10} {'Significant':>12}")
    print("-" * 57)
    for name, var in variables.items():
        result = mk_res[name]
        sig = "Yes" if result.p < alpha else "No"
        print(f"{name:<25} {result.slope:>10.3f} {result.p:>10.4f} {sig:>12}")

mk_results = run_mk(variables, alpha)
mk_results_sub = run_mk(variables_sub, alpha)

print_mk_table(mk_results, variables, f"full period ({years.min()}–{years.max()})")
print_mk_table(mk_results_sub, variables_sub, "2000–2025")

# ============================================================
# 2. Combined three-panel figure (full period)
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
axes[0].set_title(f"{site.capitalize()}: Trends in Snow-Free Days and Climate Drivers")
fig.tight_layout()
fig.savefig(fig_dir / "driver_trends_combined.png", dpi=150)
print(f"\nSaved: driver_trends_combined.png")

# ============================================================
# 3. Summary narrative (full period)
# ============================================================
def print_interpretation(mk_res, label, alpha):
    print(f"\n{'=' * 60}")
    print(f"Interpretation — {label}")
    print("=" * 60)
    sfd_sig = mk_res["Snow-free days"].p < alpha
    pdd_sig = mk_res["PDD"].p < alpha
    wp_sig  = mk_res["Winter precipitation"].p < alpha

    if not sfd_sig and not pdd_sig and not wp_sig:
        print("  No significant trends in snow-free days or either driver.")
    elif not sfd_sig and (pdd_sig or wp_sig):
        print("  No significant trend in snow-free days, but trends detected")
        print("  in one or more drivers.")
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

print_interpretation(mk_results,     f"full period ({years.min()}–{years.max()})", alpha)
print_interpretation(mk_results_sub, "2000–2025", alpha)

# ============================================================
# 4. CSV output — both periods
# ============================================================
def mk_to_rows(mk_res, variables, period_label):
    rows = []
    for name, var in variables.items():
        result = mk_res[name]
        rows.append({
            "period":    period_label,
            "variable":  name,
            "slope":     result.slope,
            "p_value":   result.p,
            "tau":       result.Tau,
            "significant": result.p < alpha,
            "slope_unit": var["slope_unit"],
        })
    return rows

full_label = f"{years.min()}-{years.max()}"
rows_full = mk_to_rows(mk_results,     variables,     full_label)
rows_sub  = mk_to_rows(mk_results_sub, variables_sub, "2000-2025")

csv_out = pd.DataFrame(rows_full + rows_sub)
csv_path = csv_dir / f"driver_trends_{site}.csv"
csv_out.to_csv(csv_path, index=False)
print(f"\nSaved CSV: {csv_path}")

# ============================================================
# 5. LaTeX table — both periods side by side
# ============================================================
tex_path = latex_dir / f"driver_trends_{site}.tex"
with open(tex_path, "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    caption = (f"Mann-Kendall trend analysis of snow-free days and climate drivers "
               f"at {site.capitalize()} for the full period "
               f"({years.min()}--{years.max()}) and 2000--2025.")
    f.write(f"\\caption{{{caption}}}\n")
    f.write(f"\\label{{tab:driver_trends}}\n")
    f.write("\\begin{tabular}{lrrrrrr}\n")
    f.write("\\hline\n")
    f.write(" & \\multicolumn{3}{c}{" + full_label.replace("-", "--") + "}"
            " & \\multicolumn{3}{c}{2000--2025} \\\\\n")
    f.write("\\cline{2-4}\\cline{5-7}\n")
    f.write("Variable & Sen's slope & $p$ & Sig. & Sen's slope & $p$ & Sig. \\\\\n")
    f.write("\\hline\n")
    for name, var in variables.items():
        r_full = mk_results[name]
        r_sub  = mk_results_sub[name]
        unit   = var["slope_unit"]
        sig_f  = "Y" if r_full.p < alpha else "N"
        sig_s  = "Y" if r_sub.p  < alpha else "N"
        f.write(f"{name} ({unit}) & "
                f"{r_full.slope:.3f} & {r_full.p:.4f} & {sig_f} & "
                f"{r_sub.slope:.3f} & {r_sub.p:.4f} & {sig_s} \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"Saved LaTeX table: {tex_path}")
print("\nDone.")
