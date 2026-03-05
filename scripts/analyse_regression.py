"""
Regression Analysis: Snow-Free Days vs Climate Predictors — Zackenberg
======================================================================
Two-predictor OLS regression:
    snow_free_days ~ PDD + winter_precip

Produces:
  - Full regression summary (statsmodels)
  - Observed vs predicted scatter plot
  - Residual diagnostics (normality, autocorrelation)
  - Time series of observed vs predicted
  - LaTeX table of regression results

Usage:
    python analyse_regression.py

Requires: statsmodels (conda install -c conda-forge statsmodels)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from pathlib import Path

# --- Config ---
workingdir = Path("/home/shl/mdrev/projects/modis/snow_cover")
csv_dir = workingdir / "results" / "csvs"
fig_dir = workingdir / "figures" / "zackenberg"
fig_dir.mkdir(parents=True, exist_ok=True)

threshold = 40
alpha = 0.05

# --- Load and merge data ---
print("Loading data...")
sfd = pd.read_csv(csv_dir / "snow_free_days.csv")
climate = pd.read_csv(csv_dir / "climate_predictors_zackenberg.csv")

df = pd.merge(sfd[["year", f"sfd_lt{threshold}"]], climate, on="year", how="inner")
df = df.dropna()
df = df.rename(columns={
    f"sfd_lt{threshold}": "snow_free_days",
    "pdd_carra": "PDD",
    "winter_precip_mm": "winter_precip",
    "melt_days": "melt_days"
})

print(f"Years: {df['year'].min()}–{df['year'].max()} (n={len(df)})")

# --- Define predictors and target ---
y = df["snow_free_days"]
X = df[["PDD", "winter_precip"]]
X_const = sm.add_constant(X)

# ============================================================
# 1. Fit OLS regression
# ============================================================
print("\n" + "=" * 60)
print("OLS Regression: snow_free_days ~ PDD + winter_precip")
print("=" * 60)

model = sm.OLS(y, X_const).fit()
print(model.summary())

# Extract key results
r2 = model.rsquared
r2_adj = model.rsquared_adj
f_pval = model.f_pvalue
residuals = model.resid
predicted = model.fittedvalues

# ============================================================
# 2. Observed vs Predicted scatter
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(predicted, y, color="steelblue", s=50, zorder=3, edgecolor="white")

# 1:1 line
lims = [min(predicted.min(), y.min()) - 5, max(predicted.max(), y.max()) + 5]
ax.plot(lims, lims, "--", color="gray", linewidth=1, label="1:1 line")

# Annotate with stats
ax.text(0.05, 0.95,
        f"R² = {r2:.3f}\nAdj. R² = {r2_adj:.3f}\nF p-value = {f_pval:.4f}\nn = {len(df)}",
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

ax.set_xlabel("Predicted snow-free days")
ax.set_ylabel("Observed snow-free days")
ax.set_title(f"Zackenberg: Observed vs Predicted Snow-Free Days\n"
             f"(PDD + Winter Precip, threshold < {threshold}%)")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(fig_dir / "regression_observed_vs_predicted.png", dpi=150)
print(f"\nSaved: regression_observed_vs_predicted.png")

# ============================================================
# 3. Time series: observed vs predicted
# ============================================================
fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(df["year"], y, "o-", color="steelblue", linewidth=1.5,
         markersize=5, label="Observed")
ax2.plot(df["year"], predicted, "s--", color="firebrick", linewidth=1.5,
         markersize=4, label="Predicted (OLS)")

# Shade residuals
ax2.fill_between(df["year"], y, predicted, alpha=0.15, color="gray",
                 label="Residual")

ax2.set_xlabel("Year")
ax2.set_ylabel(f"Snow-free days (NDSI < {threshold}%)")
ax2.set_title(f"Zackenberg: Observed vs Predicted Snow-Free Days")
ax2.legend(loc="best")
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(fig_dir / "regression_timeseries.png", dpi=150)
print(f"Saved: regression_timeseries.png")

# ============================================================
# 4. Residual diagnostics
# ============================================================
fig3, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 4a. Residuals vs fitted
axes[0].scatter(predicted, residuals, color="steelblue", s=40, edgecolor="white")
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].set_xlabel("Fitted values")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs Fitted")
axes[0].grid(True, alpha=0.3)

# 4b. QQ plot
sm.qqplot(residuals, line="s", ax=axes[1], markerfacecolor="steelblue",
          markeredgecolor="white", markersize=5)
axes[1].set_title("Q-Q Plot of Residuals")
axes[1].grid(True, alpha=0.3)

# 4c. Residuals over time (check for autocorrelation)
axes[2].plot(df["year"], residuals, "o-", color="steelblue", markersize=4)
axes[2].axhline(0, color="black", linewidth=0.5)
axes[2].set_xlabel("Year")
axes[2].set_ylabel("Residuals")
axes[2].set_title("Residuals over Time")
axes[2].grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig(fig_dir / "regression_diagnostics.png", dpi=150)
print(f"Saved: regression_diagnostics.png")

# ============================================================
# 5. Formal residual tests
# ============================================================
print("\n" + "=" * 60)
print("Residual diagnostics")
print("=" * 60)

# Shapiro-Wilk test for normality
sw_stat, sw_p = stats.shapiro(residuals)
print(f"  Shapiro-Wilk normality:  W={sw_stat:.4f}, p={sw_p:.4f} "
      f"({'normal' if sw_p > alpha else 'non-normal'})")

# Durbin-Watson test for autocorrelation
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuals)
print(f"  Durbin-Watson:           {dw:.3f} "
      f"(~2 = no autocorrelation)")

# Breusch-Pagan test for heteroscedasticity
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_const)
print(f"  Breusch-Pagan:           stat={bp_stat:.4f}, p={bp_p:.4f} "
      f"({'homoscedastic' if bp_p > alpha else 'heteroscedastic'})")

# ============================================================
# 6. Standardised coefficients (for comparing effect sizes)
# ============================================================
print("\n" + "=" * 60)
print("Standardised coefficients (beta weights)")
print("=" * 60)

X_std = (X - X.mean()) / X.std()
X_std_const = sm.add_constant(X_std)
model_std = sm.OLS((y - y.mean()) / y.std(), X_std_const).fit()

for name in ["PDD", "winter_precip"]:
    coef = model_std.params[name]
    pval = model_std.pvalues[name]
    print(f"  {name:<20} β = {coef:+.3f}  (p={pval:.4f})")

print("\n  (β weights are directly comparable: larger |β| = stronger effect)")

# ============================================================
# 7. LaTeX table
# ============================================================
tex_path = fig_dir / "regression_results.tex"
with open(tex_path, "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{OLS regression of snow-free days on PDD and winter "
            f"precipitation, Zackenberg (NDSI threshold $<$ {threshold}\\%). "
            f"n = {len(df)} years.}}\n")
    f.write("\\label{tab:regression}\n")
    f.write("\\begin{tabular}{lrrrr}\n")
    f.write("\\hline\n")
    f.write("Predictor & Coefficient & Std. Error & $t$ & $p$-value \\\\\n")
    f.write("\\hline\n")
    for name in model.params.index:
        label = name if name != "const" else "Intercept"
        label = label.replace("_", " ").replace("winter precip", "Winter precip (mm)")
        label = label.replace("PDD", "PDD (°C$\\cdot$d)")
        f.write(f"{label} & {model.params[name]:.3f} & "
                f"{model.bse[name]:.3f} & {model.tvalues[name]:.2f} & "
                f"{model.pvalues[name]:.4f} \\\\\n")
    f.write("\\hline\n")
    f.write(f"\\multicolumn{{5}}{{l}}"
            f"{{R$^2$ = {r2:.3f}, Adj.\\ R$^2$ = {r2_adj:.3f}, "
            f"F p-value = {f_pval:.4f}}} \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"\nSaved LaTeX table: {tex_path}")

# --- Export predicted values ---
df_out = df[["year", "snow_free_days"]].copy()
df_out["predicted"] = predicted.values
df_out["residual"] = residuals.values
df_out.to_csv(csv_dir / "regression_results_zackenberg.csv", index=False)
print(f"Saved: regression_results_zackenberg.csv")

print("\nDone.")
