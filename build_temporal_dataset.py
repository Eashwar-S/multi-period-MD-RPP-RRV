import os
import glob
import json
import yaml
import pandas as pd
from utils.io_utils import setup_logger, load_config
from utils.feature_utils import create_temporal_features
from utils.split_utils import balanced_chronological_split as chronological_split
from utils.reproducibility import set_seed
import pickle
from datetime import datetime

logger = setup_logger("BuildDataset", "outputs/logs/build_temporal_dataset.log")

def build_dataset(config):
    logger.info("Starting temporal dataset building pipeline...")
    set_seed(config['project']['seed'])
    
    raw_dir = config['paths']['raw_data_dir']
    out_dir = config['paths']['processed_data_dir']
    
    # 1. Load raw CSVs
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {raw_dir}")
        return
        
    logger.info(f"Found {len(csv_files)} raw CSV files. Loading & concatenating...")
    df_list = []
    for f in csv_files:
        df = pd.read_csv(f)
        # Identify area from filename if possible
        area_name = os.path.basename(f).replace("_Jan_2026.csv", "").strip()
        df['area_id'] = area_name
        df_list.append(df)
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Check if 'date' requires parse
    full_df['date'] = pd.to_datetime(full_df['date'], utc=True)
    
    # Schema checks
    required_cols = ['node_id', 'date', 'icy_label']
    for col in required_cols:
        if col not in full_df.columns:
            logger.error(f"Missing required column: {col}")
            return
            
    # Fill NaNs in features with 0 to prevent dropping entire cities' timeline
    numerical_cols = full_df.select_dtypes(include=['number']).columns
    full_df[numerical_cols] = full_df[numerical_cols].fillna(0)
            
    # Combine static and weather for lag
    # We lag everything possible
    weather_cols = config['data']['weather_cols']
    cols_to_lag = weather_cols + ['icy_base', 'severity', 'pavement_temp', 'ice_thickness']
    # keep only those that exist
    cols_to_lag = [c for c in cols_to_lag if c in full_df.columns]
    
    logger.info("Applying Temporal Feature Engineering...")
    # 2. Add temporal features per node
    df_temporal = create_temporal_features(
        full_df, 
        entity_col=config['data']['entity_col'], 
        date_col=config['data']['date_col'], 
        cols_to_lag=cols_to_lag, 
        target_col=config['data']['target_col'], 
        max_lag=config['data']['lag_days'], 
        window_size=config['data']['rolling_window']
    )
    
    # Drop rows that have NaNs due to shifting (start of the dates)
    before_len = len(df_temporal)
    df_temporal = df_temporal.dropna().reset_index(drop=True)
    after_len = len(df_temporal)
    logger.info(f"Dropped {before_len - after_len} rows due to temporal shift NaNs.")
    
    # 3. Add a placeholder split column (will dynamically split in the training scripts)
    df_temporal['split'] = 'train'
    
    # 4. Save processed tabular dataframe and split manifest
    tabular_out = os.path.join(out_dir, "processed_tabular.csv")
    df_temporal.to_csv(tabular_out, index=False)
    logger.info(f"Saved processed tabular data to {tabular_out}")
    
    split_manifest = df_temporal[['area_id', 'node_id', 'date', 'split', 'icy_label']]
    manifest_out = os.path.join(out_dir, "split_manifest.csv")
    split_manifest.to_csv(manifest_out, index=False)
    
    # List features used
    base_features = config['data']['static_cols'] + weather_cols
    # Find all lag/roll columns added
    new_temporal_cols = [c for c in df_temporal.columns if '_lag' in c or '_roll' in c or c in ['consecutive_icy_days', 'freeze_thaw']]
    
    all_features = [c for c in base_features + new_temporal_cols if c in df_temporal.columns]
    
    schema_report = {
        "entity_col": config['data']['entity_col'],
        "target_col": config['data']['target_col'],
        "temporal_features": new_temporal_cols,
        "all_model_features": all_features,
        "n_rows": len(df_temporal),
        "split_counts": df_temporal['split'].value_counts().to_dict()
    }
    
    with open(os.path.join(out_dir, "schema_report.json"), "w") as f:
        json.dump(schema_report, f, indent=4)
        
    logger.info("Dataset fully built! Ready for model training.")

if __name__ == "__main__":
    cfg = load_config()
    build_dataset(cfg)
