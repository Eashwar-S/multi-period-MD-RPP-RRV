#!/usr/bin/env python3
"""
plot_icy_distribution.py
========================
Plots a histogram of icy node counts per date for each city in graph_data/.
One subplot per city, saved as a single PNG in histogram_ground_truth_distribution/.
"""

import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR  = os.path.join(SCRIPT_DIR, 'graph_data')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'histogram_ground_truth_distribution')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'axes.facecolor':    '#F8F9FA',
    'figure.facecolor':  '#FFFFFF',
})

ACCENT   = '#3B82F6'   # blue bars
ICY_LINE = '#EF4444'   # red mean line

# ── Load all CSVs ─────────────────────────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(GRAPH_DIR, '*.csv')))
print(csv_files)
if not csv_files:
    print(f"No CSVs found in {GRAPH_DIR}")
    exit(1)

city_data: dict = {}   # city_name -> pd.Series (icy count per date)

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    # icy nodes per date
    daily_icy = df.groupby('date')['icy_label'].sum().reset_index()
    daily_icy.columns = ['date', 'icy_count']
    # city name: strip _Jan_2026.csv
    city = os.path.basename(csv_path).replace('_Jan_2026.csv', '').replace('_', ' ')
    city_data[city] = daily_icy['icy_count'].values

n_cities = len(city_data)
ncols    = 3
nrows    = int(np.ceil(n_cities / ncols))

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols * 5.5, nrows * 4),
                         constrained_layout=True)
fig.suptitle('Ground-Truth Icy Node Count Distribution per Day',
             fontsize=16, fontweight='bold', y=1.01)

axes_flat = axes.flatten() if n_cities > 1 else [axes]

for ax, (city, counts) in zip(axes_flat, city_data.items()):
    total_days  = len(counts)
    mean_val    = counts.mean()
    n_icy_days  = (counts > 0).sum()

    # Histogram
    bins = max(5, int(np.sqrt(total_days)))
    ax.hist(counts, bins=bins, color=ACCENT, edgecolor='white',
            linewidth=0.8, alpha=0.85)

    # Mean line
    ax.axvline(mean_val, color=ICY_LINE, lw=2, ls='--',
               label=f'Mean: {mean_val:.1f}')

    ax.set_title(city, fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel('Icy Nodes per Day', fontsize=9)
    ax.set_ylabel('Days (count)', fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8, framealpha=0.7)

    # Annotation box
    icy_pct = 100 * n_icy_days / total_days if total_days > 0 else 0
    ax.text(0.97, 0.97,
            f'{n_icy_days}/{total_days} icy days\n({icy_pct:.0f}%)',
            transform=ax.transAxes,
            ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='#CCCCCC'))

# Hide unused subplots
for ax in axes_flat[n_cities:]:
    ax.set_visible(False)

# Save
out_path = os.path.join(OUT_DIR, 'icy_node_distribution.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out_path}")

# ── Also save a combined summary bar chart (mean icy nodes per city) ──────────
fig2, ax2 = plt.subplots(figsize=(12, 5))
cities = list(city_data.keys())
means  = [city_data[c].mean() for c in cities]
stds   = [city_data[c].std()  for c in cities]
colors = [ACCENT if m > 0 else '#9CA3AF' for m in means]

bars = ax2.bar(range(len(cities)), means, yerr=stds,
               color=colors, edgecolor='white', linewidth=0.8,
               capsize=4, error_kw=dict(ecolor='#6B7280', lw=1.5))

ax2.set_xticks(range(len(cities)))
ax2.set_xticklabels([c.replace(', ', '\n') for c in cities],
                    fontsize=8, ha='center')
ax2.set_ylabel('Mean Icy Nodes per Day (± std)', fontsize=11)
ax2.set_title('Average Daily Icy Node Count per City',
              fontsize=13, fontweight='bold')
ax2.axhline(0, color='black', lw=0.8)
ax2.set_facecolor('#F8F9FA')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', alpha=0.3)

for i, (m, s) in enumerate(zip(means, stds)):
    ax2.text(i, m + s + 0.5, f'{m:.1f}', ha='center', va='bottom', fontsize=8)

fig2.tight_layout()
out_bar = os.path.join(OUT_DIR, 'mean_icy_nodes_per_city.png')
fig2.savefig(out_bar, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"Saved → {out_bar}")
