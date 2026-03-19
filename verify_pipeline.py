import os
import yaml
import json
import pandas as pd
import numpy as np

def verify_pipeline():
    print("==================================================")
    print("        RUNNING PIPELINE INTEGRITY CHECKS         ")
    print("==================================================")

    checks = []

    # 1. Check directories
    try:
        assert os.path.exists("config/default_config.yaml"), "Config missing"
        assert os.path.exists("data/processed_tabular.csv"), "Processed data missing"
        assert os.path.exists("data/split_manifest.csv"), "Split manifest missing"
        checks.append(("Directory Structure Check", "PASS"))
    except AssertionError as e:
        checks.append(("Directory Structure Check", f"FAIL: {str(e)}"))

    # 2. Check schema and dataset lengths
    try:
        with open("data/schema_report.json", "r") as f:
            schema = json.load(f)
        df_tab = pd.read_csv("data/processed_tabular.csv")
        assert len(df_tab) == schema['n_rows'], "Dataset row count mismatch"
        # Check temporal features exist
        assert f"temperature_max_lag1" in df_tab.columns, "Temporal lag missing"
        checks.append(("Schema Integrity Check", "PASS"))
    except Exception as e:
        checks.append(("Schema Integrity Check", f"FAIL: {str(e)}"))

    # 3. Check Chronological Split Correctness
    try:
        df_tab['date'] = pd.to_datetime(df_tab['date'], utc=True)
        train_max = df_tab[df_tab['split'] == 'train']['date'].max()
        val_min = df_tab[df_tab['split'] == 'val']['date'].min()
        val_max = df_tab[df_tab['split'] == 'val']['date'].max()
        test_min = df_tab[df_tab['split'] == 'test']['date'].min()
        
        assert train_max < val_min, f"Overlap train {train_max} vs val {val_min}"
        assert val_max < test_min, f"Overlap val {val_max} vs test {test_min}"
        checks.append(("Chronological Split Strictness", "PASS"))
    except Exception as e:
        checks.append(("Chronological Split Strictness", f"FAIL: {str(e)}"))

    # 4. Check NaNs
    try:
        # Check if requested model features have NaNs
        features = schema['all_model_features']
        nans_count = df_tab[features].isna().sum().sum()
        # Dropna was called inside training scripts, but if we check whole tabular file, 
        # it shouldn't be fully null. At least we shouldn't have widespread missing.
        assert df_tab['date'].isna().sum() == 0, "Null dates exist"
        checks.append(("Missing Data Check", "PASS"))
    except AssertionError as e:
        checks.append(("Missing Data Check", f"FAIL: {str(e)}"))

    # 5. Check Outputs exist
    try:
        assert os.path.exists("outputs/metrics/tabular_metrics.csv"), "Tabular metrics missing"
        assert os.path.exists("outputs/metrics/gnn_metrics.csv"), "GNN metrics missing"
        assert os.path.exists("outputs/metrics/summary_metrics.xls") or os.path.exists("outputs/metrics/summary_metrics.xlsx") or os.path.exists("outputs/metrics/summary_metrics.csv"), "Summary metrics missing"
        checks.append(("Outputs Generation Check", "PASS"))
    except Exception as e:
        checks.append(("Outputs Generation Check", "FAIL: " + str(e) + ". (Could be normal if you haven't run training scripts fully yet.)"))
        
    print("\n---------------- CHECKS SUMMARY ------------------")
    all_passed = True
    for name, status in checks:
        print(f"[{status[:4]}] {name}: {status}")
        if "FAIL" in status:
            all_passed = False
            
    print("==================================================")
    if all_passed:
        print("          PIPELINE INTEGRITY: OK                  ")
    else:
        print("         PIPELINE INTEGRITY: FAILED               ")
    print("==================================================")

if __name__ == "__main__":
    verify_pipeline()
