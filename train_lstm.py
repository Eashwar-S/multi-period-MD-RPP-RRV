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

logger = setup_logger("TrainLSTM", "outputs/logs/train_lstm.log")

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
    
    df_train = df[df['split'] == 'train']
    df_val = df[df['split'] == 'val']
    df_test = df[df['split'] == 'test']
    
    X_train_seq = get_sequences(df_train, base_features, lag_days)
    y_train = df_train[target].values.astype(np.float32)
    
    X_val_seq = get_sequences(df_val, base_features, lag_days)
    y_val = df_val[target].values.astype(np.float32)
    
    X_test_seq = get_sequences(df_test, base_features, lag_days)
    y_test = df_test[target].values.astype(np.float32)
    
    # Standardization across features
    # Standardize over (N * seq_len, F)
    # Fit only on train
    scaler = StandardScaler()
    X_train_flat = X_train_seq.reshape(-1, len(base_features))
    scaler.fit(X_train_flat)
    
    X_train_seq = scaler.transform(X_train_flat).reshape(X_train_seq.shape)
    X_val_seq = scaler.transform(X_val_seq.reshape(-1, len(base_features))).reshape(X_val_seq.shape)
    X_test_seq = scaler.transform(X_test_seq.reshape(-1, len(base_features))).reshape(X_test_seq.shape)
    
    # Save scaler
    with open(os.path.join(config['paths']['models_dir'], "lstm_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    logger.info(f"Train/Val/Test instances: {len(X_train_seq)} / {len(X_val_seq)} / {len(X_test_seq)}")
    
    # PyTorch DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test_seq), torch.tensor(y_test))
    
    bs = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMBaseline(input_dim=len(base_features), hidden_dim=64, num_layers=2).to(device)
    
    # Handling imbalance with pos_weight
    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / (sum(y_train) + 1e-8)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    best_val_f1 = -1
    best_val_prauc = -1
    patience = config['training']['patience']
    epochs_no_improve = 0
    
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
        
        # Validation
        model.eval()
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                probs = torch.sigmoid(logits)
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(y_b.cpu().numpy())
                
        metrics = compute_metrics(val_targets, np.array(val_probs), threshold=0.5)
        val_prauc = metrics["PR_AUC"]
        val_f1 = metrics["F1"]
        
        if val_prauc > best_val_prauc or (val_prauc == best_val_prauc and val_f1 > best_val_f1):
            best_val_prauc = val_prauc
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(config['paths']['models_dir'], "lstm_best.pt"))
        else:
            epochs_no_improve += 1
            
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val PR-AUC: {val_prauc:.4f} | Val F1: {val_f1:.4f}")
            
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
            
    # Evaluation
    model.load_state_dict(torch.load(os.path.join(config['paths']['models_dir'], "lstm_best.pt")))
    model.eval()
    
    # Tune threshold on val
    best_threshold = tune_threshold_on_validation(val_targets, np.array(val_probs), metric="F1")
    logger.info(f"Tuned Threshold: {best_threshold:.3f}")
    
    # Test
    test_probs = []
    with torch.no_grad():
        for X_b, _ in test_loader:
            X_b = X_b.to(device)
            logits = model(X_b)
            probs = torch.sigmoid(logits)
            test_probs.extend(probs.cpu().numpy())
            
    test_probs = np.array(test_probs)
    test_metrics = compute_metrics(y_test, test_probs, threshold=best_threshold)
    test_metrics["Model"] = "LSTM"
    test_metrics["Threshold"] = best_threshold
    
    pd.DataFrame([test_metrics]).to_csv(os.path.join(config['paths']['metrics_dir'], "lstm_metrics.csv"), index=False)
    logger.info(f"LSTM Test PR-AUC: {test_metrics['PR_AUC']:.4f}, F1: {test_metrics['F1']:.4f}")
    
    plot_pr_curve(y_test, test_probs, "LSTM", os.path.join(config['paths']['plots_dir'], "pr_curve_LSTM.png"))
    plot_confusion_matrix(y_test, (test_probs >= best_threshold).astype(int), "LSTM", os.path.join(config['paths']['plots_dir'], "confusion_matrix_LSTM.png"))
    
    df_test_preds = df_test[['date', 'node_id', target]].copy()
    df_test_preds['prob'] = test_probs
    df_test_preds['pred'] = (test_probs >= best_threshold).astype(int)
    df_test_preds.to_csv(os.path.join(config['paths']['predictions_dir'], "LSTM_test_predictions.csv"), index=False)
    
if __name__ == "__main__":
    train_lstm()
