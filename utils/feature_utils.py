import pandas as pd
import numpy as np

def create_temporal_features(df, entity_col='node_id', date_col='date', cols_to_lag=[], target_col='icy_label', max_lag=3, window_size=3):
    """
    Creates temporal lag and rolling features chronologically per entity.
    """
    df = df.sort_values(by=[entity_col, date_col]).reset_index(drop=True)
    
    print(f"Building lag features up to {max_lag} days and rolling windows of {window_size} days...")
    
    # Identify groups
    grouped = df.groupby(entity_col)
    
    # 1. LAG FEATURES
    # Lag the target
    for lag in range(1, max_lag + 1):
        df[f'{target_col}_lag{lag}'] = grouped[target_col].shift(lag)
        
    # Lag other features
    for col in cols_to_lag:
        for lag in range(1, max_lag + 1):
            df[f'{col}_lag{lag}'] = grouped[col].shift(lag)
            
    # 2. ROLLING FEATURES (Mean over previous days)
    # We shift(1) before rolling to avoid leaking CURRENT day's info into rolling feature
    for col in cols_to_lag:
        # e.g., rolling mean of past 3 days (not including today)
        df[f'{col}_roll{window_size}_mean'] = grouped[col].shift(1).rolling(window=window_size, min_periods=1).mean()
        
    # 3. DOMAIN-SPECIFIC TEMPORAL FEATURES
    # Consecutive icy days
    df['consecutive_icy_days'] = df.groupby(entity_col)[f'{target_col}_lag1'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
    )
    
    # Freeze-thaw indicator (went from freezing yesterday to >0 today, or vice-versa)
    if 'temperature_max' in df.columns:
        df['freeze_thaw'] = (
            ((grouped['temperature_max'].shift(1) <= 0) & (df['temperature_max'] > 0)) |
            ((grouped['temperature_max'].shift(1) > 0) & (df['temperature_max'] <= 0))
        ).astype(int)

    # Note: NaNs will be created for the first few days of each node due to shifts.
    # We leave handling of NaNs to the caller (e.g., dropping or imputing).
    
    return df
