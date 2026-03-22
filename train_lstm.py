import warnings
warnings.filterwarnings('ignore', category=UserWarning, message=".*A single label was found in 'y_true' and 'y_pred'.*")

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from utils.io_utils import setup_logger, load_config
from utils.metrics_utils import compute_metrics, tune_threshold_on_validation, plot_pr_curve, plot_confusion_matrix
from models.lstm_baseline import LSTMBaseline
from utils.reproducibility import set_seed
import pickle
from sklearn.preprocessing import StandardScaler
from datetime import datetime

logger = setup_logger("TrainLSTM", "outputs/logs/train_lstm.log")

TRAIN_AREAS = ['Arlington, Texaas, US', 'Boston, Massachusetts US', 'College Park, Maryland, US', 'Indianapolis, US']
TEST_AREAS = [
    'Little Rock, Arkansas, US', 'Louisville, Kentucky, US', 'Lubbock, Texaas, US', 
    'Memphis, US', 'Nashville, Tennessee, US', 'newyork, US', 'Oklahoma, US', 'Philadelphia, US'
]

def get_sequences(df, base_features, lag_days):
    """
    Converts flattened dataframe to (N, seq_len, F) array.
    assumes columns like 'feature', 'feature_lag1', ...
    seq_len = lag_days + 1 (current day)
    """
    N = len(df)
    F = len(base_features)
    seq_len = lag_days + 1
    
    # Pre-allocate
    X = np.zeros((N, seq_len, F), dtype=np.float32)
    
    for f_idx, base_feat in enumerate(base_features):
        # Time step 0 is lag_days ago, Time step seq_len-1 is current day
        for t in range(seq_len):
            lag_val = lag_days - t
            if lag_val == 0:
                col_name = base_feat
            else:
                col_name = f"{base_feat}_lag{lag_val}"
            
            if col_name in df.columns:
                X[:, t, f_idx] = df[col_name].values
            else:
                # Fallback if col missing
                X[:, t, f_idx] = 0.0
                
    return X

def train_lstm():
    config = load_config()
    set_seed(config['project']['seed'])
    
    data_path = os.path.join(config['paths']['processed_data_dir'], "processed_tabular.csv")
    df = pd.read_csv(data_path)
    
    target = config['data']['target_col']
    features= config['data']['static_cols'] + config['data']['weather_cols'] + ['icy_base', 'severity', 'pavement_temp', 'ice_thickness']
    base_features = [f for f in features if f in df.columns]
    
    # We must drop rows that have NaNs in any lag column used
    lag_days = config['data']['lag_days']
    all_needed_cols = []
    for f in base_features:
        all_needed_cols.append(f)
        for val in range(1, lag_days + 1):
            all_needed_cols.append(f"{f}_lag{val}")
            
    all_needed_cols = [c for c in all_needed_cols if c in df.columns]
    df = df.dropna(subset=all_needed_cols + [target])
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    df['split'] = 'unknown'
    for area in TRAIN_AREAS:
        area_mask = df['area_id'] == area
        dates = sorted(df.loc[area_mask, 'date'].unique())
        if len(dates) == 0: continue
        split_idx = max(1, int(len(dates) * 0.8))
        train_dates = dates[:split_idx]
        val_dates = dates[split_idx:]
        if len(val_dates) == 0: val_dates = train_dates
        df.loc[area_mask & df['date'].isin(train_dates), 'split'] = 'train'
        df.loc[area_mask & df['date'].isin(val_dates), 'split'] = 'val'
        
    for area in TEST_AREAS:
        df.loc[df['area_id'] == area, 'split'] = 'test'
    
    df_train_all = df[df['split'] == 'train']
    df_val_all = df[df['split'] == 'val']
    
    X_train_full = get_sequences(df_train_all, base_features, lag_days)
    y_train_full = df_train_all[target].values.astype(np.float32)
    
    scaler = StandardScaler()
    X_train_flat = X_train_full.reshape(-1, len(base_features))
    scaler.fit(X_train_flat)
    
    with open(os.path.join(config['paths']['models_dir'], "lstm_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    # Scale combined train set for actual training
    X_train_full_scaled = scaler.transform(X_train_flat).reshape(X_train_full.shape)
    train_dataset = TensorDataset(torch.tensor(X_train_full_scaled), torch.tensor(y_train_full))
    bs = 256 # Overscaling from base configuration 4 to 256 specifically for LSTM performance
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    # Create per-graph dataloaders for the 4 train areas (Train and Val)
    area_loaders = {}
    for area in TRAIN_AREAS:
        df_a_train = df_train_all[df_train_all['area_id'] == area]
        df_a_val = df_val_all[df_val_all['area_id'] == area]
        
        # sequences
        x_tr = get_sequences(df_a_train, base_features, lag_days)
        y_tr = df_a_train[target].values.astype(np.float32)
        x_va = get_sequences(df_a_val, base_features, lag_days)
        y_va = df_a_val[target].values.astype(np.float32)
        
        if len(x_tr) > 0:
            x_tr = scaler.transform(x_tr.reshape(-1, len(base_features))).reshape(x_tr.shape)
        if len(x_va) > 0:
            x_va = scaler.transform(x_va.reshape(-1, len(base_features))).reshape(x_va.shape)
            
        area_loaders[area] = {
            'train': DataLoader(TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr)), batch_size=bs, shuffle=False) if len(x_tr)>0 else None,
            'val': DataLoader(TensorDataset(torch.tensor(x_va), torch.tensor(y_va)), batch_size=bs, shuffle=False) if len(x_va)>0 else None,
            'val_targets': y_va
        }
        
    # Global validation loader for early stopping
    X_val_full = get_sequences(df_val_all, base_features, lag_days)
    y_val_full = df_val_all[target].values.astype(np.float32)
    if len(X_val_full) > 0:
        X_val_full_scaled = scaler.transform(X_val_full.reshape(-1, len(base_features))).reshape(X_val_full.shape)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val_full_scaled), torch.tensor(y_val_full)), batch_size=bs, shuffle=False)
    else:
        val_loader = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMBaseline(input_dim=len(base_features), hidden_dim=48, num_layers=2, dropout=0.5).to(device)
    
    pos_weight = torch.tensor([(len(y_train_full) - sum(y_train_full)) / (sum(y_train_full) + 1e-8)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)
    
    best_val_f1 = -1
    best_val_prauc = -1
    patience = config['training']['patience']
    epochs_no_improve = 0
    
    epoch_logs = []
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_b.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation per area
        model.eval()
        epoch_record = {"Epoch": epoch, "Global_Train_Loss": train_loss}
        
        with torch.no_grad():
            for area in TRAIN_AREAS:
                # Train loss per area
                a_tr_loss = 0.0
                if area_loaders[area]['train'] is not None:
                    count = 0
                    for X_b, y_b in area_loaders[area]['train']:
                        logits = model(X_b.to(device))
                        loss = criterion(logits, y_b.to(device))
                        a_tr_loss += loss.item() * X_b.size(0)
                        count += X_b.size(0)
                    a_tr_loss /= max(count, 1)
                epoch_record[f"{area}_Train_Loss"] = a_tr_loss
                
                # Val metrics per area
                a_va_loss = 0.0
                a_va_probs = []
                if area_loaders[area]['val'] is not None:
                    count = 0
                    for X_b, y_b in area_loaders[area]['val']:
                        logits = model(X_b.to(device))
                        loss = criterion(logits, y_b.to(device))
                        a_va_loss += loss.item() * X_b.size(0)
                        count += X_b.size(0)
                        a_va_probs.extend(torch.sigmoid(logits).cpu().numpy())
                    a_va_loss /= max(count, 1)
                    
                    a_va_metrics = compute_metrics(area_loaders[area]['val_targets'], np.array(a_va_probs))
                    epoch_record[f"{area}_Val_Loss"] = a_va_loss
                    epoch_record[f"{area}_Val_PRAUC"] = a_va_metrics["PR_AUC"]
                    epoch_record[f"{area}_Val_F1"] = a_va_metrics["F1"]
                else:
                    epoch_record[f"{area}_Val_Loss"] = np.nan
                    epoch_record[f"{area}_Val_PRAUC"] = np.nan
                    epoch_record[f"{area}_Val_F1"] = np.nan
                    
        epoch_logs.append(epoch_record)
        
        # Global Validation for early stopping
        val_probs, val_targets = [], []
        if val_loader is not None:
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    logits = model(X_b.to(device))
                    val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                    val_targets.extend(y_b.cpu().numpy())
            metrics = compute_metrics(val_targets, np.array(val_probs), threshold=0.5)
            val_prauc = metrics["PR_AUC"]
            val_f1 = metrics["F1"]
        else:
            val_prauc = 0
            val_f1 = 0
        
        if val_prauc > best_val_prauc or (val_prauc == best_val_prauc and val_f1 > best_val_f1):
            best_val_prauc = val_prauc
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(config['paths']['models_dir'], "lstm_best.pt"))
        else:
            epochs_no_improve += 1
            
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:03d} | Global Train Loss: {train_loss:.4f} | Global Val PR-AUC: {val_prauc:.4f} | Global Val F1: {val_f1:.4f}")
            
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
            
    # Save loss curves
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(epoch_logs).to_excel(os.path.join(config['paths']['metrics_dir'], f"lstm_train_curves_{timestamp}.xlsx"), index=False)
            
    # Evaluation
    model.load_state_dict(torch.load(os.path.join(config['paths']['models_dir'], "lstm_best.pt")))
    model.eval()
    
    if val_loader is not None and len(val_targets) > 0:
        best_threshold = tune_threshold_on_validation(val_targets, np.array(val_probs), metric="F1")
    else:
        best_threshold = 0.5
    logger.info(f"Tuned Threshold: {best_threshold:.3f}")
    
    df_test_all = df[(df['area_id'].isin(TEST_AREAS))]
    test_metrics_list = []
    
    for area in TEST_AREAS:
        df_a_test = df_test_all[df_test_all['area_id'] == area]
        x_te = get_sequences(df_a_test, base_features, lag_days)
        y_te = df_a_test[target].values.astype(np.float32)
        
        if len(x_te) == 0:
            continue
            
        x_te_scaled = scaler.transform(x_te.reshape(-1, len(base_features))).reshape(x_te.shape)
        te_loader = DataLoader(TensorDataset(torch.tensor(x_te_scaled), torch.tensor(y_te)), batch_size=bs, shuffle=False)
        
        te_probs = []
        with torch.no_grad():
            for X_b, _ in te_loader:
                logits = model(X_b.to(device))
                te_probs.extend(torch.sigmoid(logits).cpu().numpy())
                
        te_probs = np.array(te_probs)
        metrics = compute_metrics(y_te, te_probs, threshold=best_threshold)
        metrics["Model"] = "LSTM"
        metrics["Area"] = area
        test_metrics_list.append(metrics)
        
    if len(test_metrics_list) > 0:
        df_test_metrics = pd.DataFrame(test_metrics_list)
        df_test_metrics.to_excel(os.path.join(config['paths']['metrics_dir'], f"lstm_test_metrics_{timestamp}.xlsx"), index=False)
        
        avg_prauc = df_test_metrics['PR_AUC'].mean()
        avg_f1 = df_test_metrics['F1'].mean()
        logger.info(f"LSTM Test Average PR-AUC: {avg_prauc:.4f}, Average F1: {avg_f1:.4f}")
    else:
        logger.warning("LSTM Test metrics list is empty.")
    
if __name__ == "__main__":
    train_lstm()
