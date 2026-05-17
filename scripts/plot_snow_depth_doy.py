"""
Plot in situ snow depth by day of year, one line per year.

Usage:
    python scripts/plot_snow_depth_doy.py --site disko
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True)
args = parser.parse_args()

with open(f"config/{args.site}.yml") as f:
    cfg = yaml.safe_load(f)

site        = cfg["site"]
gem_dir     = Path(cfg["gem_dir"])
sd_file     = cfg["insitu_snowdepth"]
col_sd      = cfg.get("insitu_col_snowdepth",      "SD (m)")
time_col_sd = cfg.get("insitu_time_col_snowdepth", "Time")

fig_dir = Path(f"figures/{site}/")
fig_dir.mkdir(parents=True, exist_ok=True)

# --- Load ---
sd_path = next(gem_dir.glob(sd_file))
df = pd.read_csv(sd_path, sep="\t", na_values=-9999, encoding="utf-8-sig")
df["datetime"] = pd.to_datetime(df["Date"] + " " + df[time_col_sd])
df = df.set_index("datetime")
sd_daily = df[col_sd].resample("1D").mean()

years = sorted(sd_daily.index.year.unique())
cmap  = cm.viridis
colors = {yr: cmap(i / max(len(years) - 1, 1)) for i, yr in enumerate(years)}

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))

for year, grp in sd_daily.groupby(sd_daily.index.year):
    doy = grp.index.dayofyear
    ax.plot(doy, grp.values, color=colors[year], linewidth=0.9, alpha=0.85,
            label=str(year))

ax.set_xlabel("Day of year", fontsize=10)
ax.set_ylabel("Snow depth (m)", fontsize=10)
ax.set_title(f"{site.capitalize()}: daily snow depth by year", fontsize=11)
ax.set_xlim(1, 366)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

# Colourbar as year legend
sm = cm.ScalarMappable(cmap=cmap,
                       norm=plt.Normalize(vmin=min(years), vmax=max(years)))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01)
cbar.set_label("Year", fontsize=9)
cbar.set_ticks(np.linspace(min(years), max(years), min(len(years), 8)).astype(int))

fig.tight_layout()
out = fig_dir / "snow_depth_by_doy.png"
fig.savefig(out, dpi=300)
print(f"Saved: {out}")
