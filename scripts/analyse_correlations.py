"""
Correlation Analysis: Snow-Free Days vs Climate Predictors — Zackenberg
=======================================================================
Loads snow-free days and climate predictors from CSVs and computes:
  1. Pairwise Pearson correlation matrix with significance
  2. Partial correlations (each predictor controlling for the others)
  3. Visualisations: correlation heatmap and partial correlation bar chart

Usage:
    python analyse_correlations.py

Requires: pingouin (conda install -c conda-forge pingouin)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pingouin as pg
from pathlib import Path

# --- Config ---
workingdir = Path("/home/shl/mdrev/projects/modis/snow_cover")
csv_dir = workingdir / "results" / "csvs"
fig_dir = workingdir / "figures" / "zackenberg"
fig_dir.mkdir(parents=True, exist_ok=True)

threshold = 40  # NDSI threshold for snow-free days
alpha = 0.05

# --- Load data ---
print("Loading data...")
sfd = pd.read_csv(csv_dir / "snow_free_days.csv")
climate = pd.read_csv(csv_dir / "climate_predictors_zackenberg.csv")

# Merge on year
df = pd.merge(sfd[["year", f"sfd_lt{threshold}"]], climate, on="year", how="inner")
df = df.dropna()
df = df.rename(columns={
    f"sfd_lt{threshold}": "snow_free_days",
    "pdd_carra": "PDD",
    "winter_precip_mm": "Winter Precip",
    "melt_days": "Melt Days"
})

print(f"Overlapping years: {df['year'].min()}–{df['year'].max()} ({len(df)} years)")
print(f"\n{df.describe().round(1).to_string()}")

# Variables for analysis
target = "snow_free_days"
predictors = ["PDD", "Winter Precip", "Melt Days"]
all_vars = [target] + predictors

# ============================================================
# 1. Pairwise Pearson correlation matrix with p-values
# ============================================================
print("\n" + "=" * 60)
print("Pairwise Pearson correlations")
print("=" * 60)

n = len(all_vars)
corr_matrix = np.zeros((n, n))
pval_matrix = np.zeros((n, n))

for i, v1 in enumerate(all_vars):
    for j, v2 in enumerate(all_vars):
        r, p = stats.pearsonr(df[v1], df[v2])
        corr_matrix[i, j] = r
        pval_matrix[i, j] = p

corr_df = pd.DataFrame(corr_matrix, index=all_vars, columns=all_vars).round(3)
pval_df = pd.DataFrame(pval_matrix, index=all_vars, columns=all_vars).round(4)

print("\nCorrelation coefficients:")
print(corr_df.to_string())
print("\np-values:")
print(pval_df.to_string())

# --- Heatmap ---
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(all_vars, rotation=45, ha="right")
ax.set_yticklabels(all_vars)

# Annotate with r and significance stars
for i in range(n):
    for j in range(n):
        r = corr_matrix[i, j]
        p = pval_matrix[i, j]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        color = "white" if abs(r) > 0.6 else "black"
        ax.text(j, i, f"{r:.2f}{stars}", ha="center", va="center",
                color=color, fontsize=10)

plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
ax.set_title("Zackenberg: Correlation Matrix\n"
             f"(Snow-free days threshold < {threshold}%)")
fig.tight_layout()
fig.savefig(fig_dir / "correlation_matrix.png", dpi=150)
print(f"\nSaved: correlation_matrix.png")

# ============================================================
# 2. Partial correlations
# ============================================================
print("\n" + "=" * 60)
print("Partial correlations (each predictor vs snow-free days,")
print("controlling for the other predictors)")
print("=" * 60)

partial_results = []
for pred in predictors:
    covars = [p for p in predictors if p != pred]
    pc = pg.partial_corr(data=df, x=target, y=pred, covar=covars)
    r = pc["r"].values[0]
    p = pc["p_val"].values[0]
    sig = "Yes" if p < alpha else "No"
    partial_results.append({
        "predictor": pred,
        "partial_r": r,
        "p_value": p,
        "significant": sig,
        "controlling_for": ", ".join(covars)
    })
    print(f"  {pred:<15} r={r:+.3f}  p={p:.4f}  {'*' if p < alpha else ''}")

partial_df = pd.DataFrame(partial_results)

# --- Bar chart ---
fig2, ax2 = plt.subplots(figsize=(8, 5))

colors = ["firebrick" if p < alpha else "lightgray"
          for p in partial_df["p_value"]]

bars = ax2.bar(partial_df["predictor"], partial_df["partial_r"],
               color=colors, edgecolor="black", linewidth=0.5)

ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("Partial correlation (r)")
ax2.set_title(f"Zackenberg: Partial Correlations with Snow-Free Days\n"
              f"(threshold < {threshold}%, controlling for other predictors)")
ax2.grid(True, alpha=0.3, axis="y")

# Annotate bars
for bar, row in zip(bars, partial_df.itertuples()):
    label = f"r={row.partial_r:+.2f}\np={row.p_value:.3f}"
    y = bar.get_height()
    va = "bottom" if y >= 0 else "top"
    ax2.text(bar.get_x() + bar.get_width() / 2, y, label,
             ha="center", va=va, fontsize=9)

# Legend for significance
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="firebrick", edgecolor="black",
                         label=f"Significant (p < {alpha})"),
                   Patch(facecolor="lightgray", edgecolor="black",
                         label="Not significant")]
ax2.legend(handles=legend_elements, loc="best")

fig2.tight_layout()
fig2.savefig(fig_dir / "partial_correlations.png", dpi=150)
print(f"\nSaved: partial_correlations.png")

# ============================================================
# 3. VIF (Variance Inflation Factors) for the predictors
# ============================================================
print("\n" + "=" * 60)
print("Variance Inflation Factors (VIF)")
print("=" * 60)

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X = add_constant(df[predictors])
vif_data = []
for i, pred in enumerate(X.columns):
    vif_data.append({"predictor": pred,
                     "VIF": variance_inflation_factor(X.values, i)})

vif_df = pd.DataFrame(vif_data)
print(vif_df.to_string(index=False))
print("\n  VIF > 5 suggests problematic multicollinearity")
print("  VIF > 10 suggests severe multicollinearity")

# ============================================================
# 4. Export results
# ============================================================
# Correlation matrix
corr_df.to_csv(csv_dir / "correlation_matrix_zackenberg.csv")

# Partial correlations
partial_df.to_csv(csv_dir / "partial_correlations_zackenberg.csv", index=False)

# LaTeX table: partial correlations
tex_path = fig_dir / "partial_correlations.tex"
with open(tex_path, "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Partial correlations between snow-free days and climate "
            f"predictors, Zackenberg (NDSI threshold $<$ {threshold}\\%). "
            "Each predictor is correlated with snow-free days while controlling "
            "for the remaining predictors.}\n")
    f.write("\\label{tab:partial_corr}\n")
    f.write("\\begin{tabular}{lrrl}\n")
    f.write("\\hline\n")
    f.write("Predictor & Partial $r$ & $p$-value & Significant \\\\\n")
    f.write("\\hline\n")
    for _, row in partial_df.iterrows():
        f.write(f"{row['predictor']} & {row['partial_r']:.3f} & "
                f"{row['p_value']:.4f} & {row['significant']} \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"\nSaved LaTeX table: {tex_path}")
print("\nDone.")
