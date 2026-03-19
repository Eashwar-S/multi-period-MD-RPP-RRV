import pandas as pd
import numpy as np


def chronological_split(df, date_col='date', entity_col='node_id', train_ratio=0.7, val_ratio=0.15):
    """
    Strict chronological split: earliest dates to train, middle to val, latest to test.
    Simple and leakage-free but may produce poor splits when icy days cluster in one period.
    """
    df = df.sort_values(by=[date_col, entity_col]).reset_index(drop=True)
    unique_dates = sorted(df[date_col].unique())
    n_dates = len(unique_dates)
    train_idx = int(n_dates * train_ratio)
    val_idx = int(n_dates * (train_ratio + val_ratio))
    train_dates = unique_dates[:train_idx]
    val_dates = unique_dates[train_idx:val_idx]
    test_dates = unique_dates[val_idx:]
    print(f"Chronological split over {n_dates} dates:")
    print(f"  Train: {len(train_dates)} dates ({train_dates[0].date()} to {train_dates[-1].date()})")
    print(f"  Val:   {len(val_dates)} dates ({val_dates[0].date()} to {val_dates[-1].date()})")
    print(f"  Test:  {len(test_dates)} dates ({test_dates[0].date()} to {test_dates[-1].date()})")
    df['split'] = 'train'
    df.loc[df[date_col].isin(val_dates), 'split'] = 'val'
    df.loc[df[date_col].isin(test_dates), 'split'] = 'test'
    assert len(set(train_dates).intersection(set(val_dates))) == 0
    assert len(set(val_dates).intersection(set(test_dates))) == 0
    return df


def balanced_chronological_split(df, date_col='date', entity_col='node_id', 
                                  target_col='icy_label',
                                  val_days=2, test_days=3):
    """
    Selects val and test date windows from INSIDE the dataset so that each split
    sees both icy (label=1) and non-icy (label=0) days if possible.

    Strategy:
      - Find dates that are icy transition days (neither all-zero nor all-one fraction)
      - Insert val window just before the largest transition in icy fraction
      - Insert test window just after the transition
      - Assign all remaining dates to train
    
    Falls back to strict chronological split if not enough transition days exist.
    """
    df = df.sort_values(by=[date_col, entity_col]).reset_index(drop=True)
    unique_dates = sorted(df[date_col].unique())
    
    # Get fraction of icy nodes per date
    icy_frac = df.groupby(date_col)[target_col].mean()
    icy_frac = icy_frac.reindex(unique_dates)
    fracs = icy_frac.values
    
    # Find the date with the biggest day-to-day change (transition point)
    transitions = np.abs(np.diff(fracs))
    if len(transitions) < val_days + test_days + 2:
        print("[Split] Not enough dates for balanced split. Falling back to chronological.")
        return chronological_split(df, date_col, entity_col)
    
    # Peak transition index (biggest jump in icy_frac, use second-largest if there's overlap potential)
    sorted_transitions = np.argsort(transitions)[::-1]
    peak_idx = sorted_transitions[0]
    
    # Test window: places test_days days starting just AFTER the peak transition
    test_start = min(peak_idx + 1, len(unique_dates) - test_days - val_days)
    test_end = test_start + test_days
    test_dates = set(unique_dates[test_start:test_end])
    
    # Val window: places val_days days just BEFORE test_start
    val_end = test_start
    val_start = max(0, val_end - val_days)
    val_dates = set(unique_dates[val_start:val_end])
    
    # All remaining dates -> train
    train_dates = set(unique_dates) - val_dates - test_dates
    
    # Verify no overlap
    assert len(train_dates.intersection(val_dates)) == 0
    assert len(val_dates.intersection(test_dates)) == 0
    
    # Check both splits have both classes
    train_icy_days = sum(1 for d in train_dates if icy_frac[d] > 0)
    val_icy_days = sum(1 for d in val_dates if icy_frac[d] > 0)
    test_icy_days = sum(1 for d in test_dates if icy_frac[d] > 0)
    
    print(f"Balanced chronological split:")
    print(f"  Train: {len(train_dates)} dates, {train_icy_days} have icy roads")
    print(f"  Val:   {len(val_dates)} dates, {val_icy_days} have icy roads")
    print(f"  Test:  {len(test_dates)} dates, {test_icy_days} have icy roads")
    
    df['split'] = df[date_col].apply(
        lambda d: 'val' if d in val_dates else ('test' if d in test_dates else 'train')
    )
    return df
