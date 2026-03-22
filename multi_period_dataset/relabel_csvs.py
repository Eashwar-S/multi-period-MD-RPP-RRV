#!/usr/bin/env python3
"""
relabel_csvs.py
================
Re-applies improved ice labeling logic to all existing graph_data/ CSVs.
No API calls needed — all required weather columns are already present.

New labeling improvements over original:
  1. strong_thaw  : requires BOTH temp_max > 4°C AND temp_min > -1°C
                    (prevents false "full thaw" when nights still drop to -4°C)
  2. partial_thaw : temp_max > 1°C AND temp_min ≤ -1°C
                    (freeze-thaw cycle — creates black ice when prev day was icy)
  3. black_ice_risk: partial_thaw + prev_icy_base = black ice hazard treated as icy
  4. icy_persist  : ice survives all days without strong_thaw
                    (partial_thaw days included since overnight refreeze occurs)
  5. icy_label    : icy_base OR icy_persist OR black_ice_risk

After running this script:
  1. Re-run build_temporal_dataset.py   → rebuilds processed_tabular.csv
  2. Re-run train_gnn_temporal.py       → retrains GNN on new labels
  3. Re-run generate_gnn_routing_instances.py  → regenerates routing bundles
"""

import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ── Path setup ─────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR    = os.path.join(SCRIPT_DIR, 'graph_data')


def relabel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply improved labeling logic to a DataFrame that already contains
    all raw weather columns.  Overwrites icy_base, strong_thaw, icy_persist,
    icy_label and adds partial_thaw, black_ice_risk, thaw_fraction.
    """
    df = df.copy()

    # ── Re-derive helper columns (same as original extract_weather_data) ──────
    df["wet_flag"] = (
        (df["precipitation_sum"] > 0) |
        (df["rain_sum"]          > 0) |
        (df["snowfall_sum"]      > 0) |
        (df["humidity_mean"]     >= 75)
    ).astype(int)

    df["severity"] = (
        (df["temperature_min"] <= 0).astype(int) +
        (df["temperature_min"] <= df["dewpoint_mean"]).astype(int) +
        (df["humidity_mean"]   >= 75).astype(int) +
        df["wet_flag"]
    )

    df["delta_local"] = np.select(
        [df["street_count"] <= 2, df["street_count"] == 3, df["street_count"] >= 4],
        [0.0, 0.3, 0.6], default=0.0
    )

    df["eligible"] = np.select(
        [df["severity"] <= 1, df["severity"] == 2, df["severity"] >= 3],
        [0, (df["street_count"] >= 4).astype(int), (df["street_count"] >= 3).astype(int)],
        default=0
    )

    df["pavement_temp"] = df["temperature_min"] - 2.0 - df["delta_local"]

    df["icy_base"] = (
        (df["eligible"]  == 1) &
        (df["pavement_temp"] <= 0) &
        (df["pavement_temp"] <= df["dewpoint_mean"]) &
        (df["wet_flag"]  == 1)
    ).astype(int)

    df["prev_icy_base"] = (
        df.groupby("node_id")["icy_base"].shift(1).fillna(0).astype(int)
    )

    # ── IMPROVED thaw logic ───────────────────────────────────────────────────

    # STRONG thaw: day warms AND night stays above -1°C (no overnight refreeze)
    # Original was ONLY temp_max > 4°C — now we add the overnight condition.
    df["strong_thaw"] = (
        (df["temperature_max"] > 4.0) &
        (df["temperature_min"] > -1.0)    # ← NEW: prevent false full-thaw
    ).astype(int)

    # PARTIAL thaw: daytime melting but sub-zero overnight → classic black ice
    # (meltwater refreezes during the night into a thin invisible film)
    df["partial_thaw"] = (
        (df["temperature_max"] > 1.0) &   # some surface melting
        (df["temperature_min"] <= -1.0)   # but overnight refreeze
    ).astype(int)

    # Thaw fraction: how much of yesterday's ice survives today
    #   1.0 = full persistence (cold day, no thaw)
    #   0.5 = partial persistence (partial thaw / refreeze cycle)
    #   0.0 = fully thawed
    df["thaw_fraction"] = np.select(
        [
            df["strong_thaw"]  == 1,   # complete thaw
            df["partial_thaw"] == 1,   # partial thaw — 50% remains and refreezes
        ],
        [0.0, 0.5],
        default=1.0                    # no thaw: 100% persists
    )

    # Black ice risk: partial thaw cycle after a previously icy day
    # (the most operationally dangerous scenario — thin invisible ice)
    df["black_ice_risk"] = (
        (df["partial_thaw"]  == 1) &
        (df["prev_icy_base"] == 1)
    ).astype(int)

    # Persistence: ice remains if thaw_fraction > 0.3 (survives no-thaw AND partial-thaw)
    df["icy_persist"] = (
        (df["prev_icy_base"]  == 1) &
        (df["thaw_fraction"]   > 0.3)
    ).astype(int)

    # Final label: new ice OR persisted ice OR black ice risk
    df["icy_label"] = (
        (df["icy_base"]       == 1) |
        (df["icy_persist"]    == 1) |
        (df["black_ice_risk"] == 1)
    ).astype(int)

    return df


def main():
    csv_files = sorted(glob.glob(os.path.join(GRAPH_DIR, '*.csv')))
    if not csv_files:
        print(f"No CSVs found in {GRAPH_DIR}")
        return

    print(f"Found {len(csv_files)} CSV files to relabel.\n")

    for csv_path in csv_files:
        name = os.path.basename(csv_path)
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'], utc=True)

        # --- Stats before ---
        before_icy = df.groupby('date')['icy_label'].sum()
        n_before   = before_icy.sum()

        # --- Relabel ---
        df = relabel(df)

        # --- Stats after ---
        after_icy = df.groupby('date')['icy_label'].sum()
        n_after   = after_icy.sum()

        # --- Save ---
        df.to_csv(csv_path, index=False)

        # --- Report ---
        print(f"{name}")
        print(f"  icy_label total:  {n_before:4.0f} → {n_after:4.0f}  "
              f"(Δ = {n_after - n_before:+.0f})")
        # Show per-date breakdown for comparison
        merged = pd.DataFrame({'before': before_icy, 'after': after_icy}).fillna(0).astype(int)
        for date, row in merged.iterrows():
            if row['before'] != row['after']:
                d_str = pd.Timestamp(date).strftime('%b %d')
                print(f"    {d_str}: {row['before']:3d} → {row['after']:3d} nodes icy")
        print()

    print("Done. Next steps:")
    print("  1. python build_temporal_dataset.py     # rebuilds processed_tabular.csv")
    print("  2. python train_gnn_temporal.py          # retrains GNN on new labels")
    print("  3. cd multi_period_routing && python generate_gnn_routing_instances.py")


if __name__ == '__main__':
    main()
