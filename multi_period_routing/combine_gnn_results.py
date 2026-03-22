#!/usr/bin/env python3
"""
combine_gnn_results.py
=======================
Run AFTER all 8 per-area jobs finish.
Walks gnn_routing_results/<area_slug>/ subdirectories, merges every
*_results.xlsx file into combined outputs in gnn_routing_results/.

Usage:
    python combine_gnn_results.py
"""

import os
import glob
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'gnn_routing_results')

# ── Collect all per-area result files ─────────────────────────────────────────
results_files = sorted(glob.glob(
    os.path.join(RESULTS_DIR, '*', '*_results.xlsx')
))

if not results_files:
    print(f"No result files found under {RESULTS_DIR}/")
    print("Make sure all 8 jobs have finished before combining.")
    exit(1)

print(f"Found {len(results_files)} result file(s):")
for f in results_files:
    print(f"  {f}")

# ── Merge raw results ─────────────────────────────────────────────────────────
dfs = []
for f in results_files:
    try:
        df = pd.read_excel(f)
        dfs.append(df)
        print(f"  Loaded {len(df)} rows from {os.path.basename(f)}")
    except Exception as e:
        print(f"  WARNING: Could not load {f}: {e}")

if not dfs:
    print("No data to combine. Exiting.")
    exit(1)

df_all = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows after merge: {len(df_all)}")
print(f"Areas present: {sorted(df_all['area'].unique())}")
print(f"Modes present: {sorted(df_all['mode'].unique())}")

# ── Save combined raw results ─────────────────────────────────────────────────
all_xlsx = os.path.join(RESULTS_DIR, 'combined_all_results.xlsx')
df_all.to_excel(all_xlsx, index=False)
print(f"\nCombined raw results saved to:\n  {all_xlsx}")

# ── Summary: mean ± std by (area, mode, period) ───────────────────────────────
summary_cols = ['obj', 'time_s', 'unique_edges_covered', '%_covered']
summary_cols = [c for c in summary_cols if c in df_all.columns]

summary_df = (
    df_all.groupby(['area', 'mode', 'period'])[summary_cols]
    .agg(['mean', 'std'])
    .reset_index()
)
summary_df.columns = ['_'.join(c).strip('_') for c in summary_df.columns.values]
summary_xlsx = os.path.join(RESULTS_DIR, 'combined_summary.xlsx')
summary_df.to_excel(summary_xlsx, index=False)
print(f"Summary (mean±std per period) saved to:\n  {summary_xlsx}")

# ── Area × mode aggregate ─────────────────────────────────────────────────────
area_mode_df = (
    df_all.groupby(['area', 'mode'])[summary_cols]
    .agg(['mean', 'std'])
    .reset_index()
)
area_mode_df.columns = ['_'.join(c).strip('_') for c in area_mode_df.columns.values]
area_mode_xlsx = os.path.join(RESULTS_DIR, 'combined_area_mode_summary.xlsx')
area_mode_df.to_excel(area_mode_xlsx, index=False)
print(f"Area×mode aggregate saved to:\n  {area_mode_xlsx}")

# ── Console pivot: objective by (area, mode) ──────────────────────────────────
pivot_obj = (
    df_all.groupby(['area', 'mode'])['obj']
    .mean()
    .unstack('mode')
)
pivot_cov = (
    df_all.groupby(['area', 'mode'])['%_covered']
    .mean()
    .unstack('mode')
)

print("\n── Mean Objective by Area × Mode ──────────────────────────────────────")
print(pivot_obj.round(4).to_string())
print("\n── Mean Coverage (%) by Area × Mode ───────────────────────────────────")
print(pivot_cov.round(2).to_string())

# ── Per-area sheets in one workbook ──────────────────────────────────────────
multi_xlsx = os.path.join(RESULTS_DIR, 'combined_per_area_sheets.xlsx')
with pd.ExcelWriter(multi_xlsx, engine='openpyxl') as writer:
    df_all.to_excel(writer, sheet_name='All', index=False)
    for area in sorted(df_all['area'].unique()):
        sheet_name = area[:31]   # Excel tab limit
        df_all[df_all['area'] == area].to_excel(writer, sheet_name=sheet_name, index=False)
print(f"\nPer-area workbook saved to:\n  {multi_xlsx}")
print("\nDone.")
