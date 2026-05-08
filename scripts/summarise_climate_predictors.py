"""
Summary statistics for climate predictors across GEM sites
==========================================================
Computes mean ± std of PDD and winter precipitation for each site
over the common MODIS period (2001–2024) and writes a LaTeX table.

Usage:
    python scripts/summarise_climate_predictors.py
"""
import pandas as pd
from pathlib import Path

workingdir = Path("/home/shl/mdrev/projects/modis/snow_cover")
latex_dir  = workingdir / "results/latex"
latex_dir.mkdir(parents=True, exist_ok=True)

SITES = ["zackenberg", "nuuk", "disko"]
PERIOD = (2001, 2024)

rows = []
for site in SITES:
    df = pd.read_csv(workingdir / f"results/csvs/{site}/climate_predictors_{site}.csv")
    df = df[(df["year"] >= PERIOD[0]) & (df["year"] <= PERIOD[1])].dropna(subset=["pdd_carra", "winter_precip_mm"])
    rows.append({
        "site":            site.capitalize(),
        "n":               len(df),
        "pdd_mean":        df["pdd_carra"].mean(),
        "pdd_std":         df["pdd_carra"].std(),
        "wp_mean":         df["winter_precip_mm"].mean(),
        "wp_std":          df["winter_precip_mm"].std(),
    })
    print(f"{site}: n={len(df)}, PDD={df['pdd_carra'].mean():.0f}±{df['pdd_carra'].std():.0f} °C·d, "
          f"winter precip={df['winter_precip_mm'].mean():.0f}±{df['winter_precip_mm'].std():.0f} mm")

summary = pd.DataFrame(rows)

# --- LaTeX table ---
tex_path = latex_dir / "climate_predictors_summary.tex"
with open(tex_path, "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Mean $\\pm$ standard deviation of annual positive degree days (PDD) "
            "and winter precipitation at the three GEM sites, "
            f"{PERIOD[0]}--{PERIOD[1]}.}}\n")
    f.write("\\label{tab:climate_summary}\n")
    f.write("\\begin{tabular}{lrr}\n")
    f.write("\\hline\n")
    f.write("Site & PDD (\\textdegree C$\\cdot$d) & Winter precipitation (mm) \\\\\n")
    f.write("\\hline\n")
    for _, row in summary.iterrows():
        f.write(f"{row['site']} & "
                f"${row['pdd_mean']:.0f} \\pm {row['pdd_std']:.0f}$ & "
                f"${row['wp_mean']:.0f} \\pm {row['wp_std']:.0f}$ \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"\nSaved LaTeX table: {tex_path}")
