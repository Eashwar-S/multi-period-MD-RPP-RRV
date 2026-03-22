"""
train_gnn_temporal.py

Trains two GNN variants on the temporal road-icing graph dataset:
  1. TemporalEdgeGNN  - GRU over sequential daily snapshots, then GNN spatial conv, then edge head
  2. StaticEdgeGNN    - Ablation: GNN directly on flattened temporal features, no recurrence

The TemporalEdgeGNN explicitly models the temporal sequence by stacking K daily snapshots
of base features as a sequence, passing them through a GRU per node, then applying graph
convolution on the final temporal hidden state before edge-level classification.

The difference versus LSTM/tabular:
  - Graph structure (edges, neighborhood) is explicitly used during spatial aggregation
  - Neighbor information is propagated AFTER temporal aggregation, so each node's final
    hidden representation is aware of its own history AND spatially informed by neighbors
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message=".*A single label was found in 'y_true' and 'y_pred'.*")

import os
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from utils.io_utils import setup_logger, load_config
from utils.metrics_utils import compute_metrics, tune_threshold_on_validation, plot_pr_curve, plot_confusion_matrix
from utils.graph_utils import build_static_graph_structure
from models.temporal_gnn import TemporalNodeGNN, StaticNodeGNN
from utils.reproducibility import set_seed
from sklearn.preprocessing import StandardScaler
from datetime import datetime

logger = setup_logger("TrainGNN", "outputs/logs/train_gnn.log")

TRAIN_AREAS = ['Arlington, Texaas, US', 'Boston, Massachusetts US', 'College Park, Maryland, US', 'Indianapolis, US']
TEST_AREAS = [
    'Little Rock, Arkansas, US', 'Louisville, Kentucky, US', 'Lubbock, Texaas, US', 
    'Memphis, US', 'Nashville, Tennessee, US', 'newyork, US', 'Oklahoma, US', 'Philadelphia, US'
]


def build_temporal_graph_sequence(df, node2idx, base_features, lag_days, date_col, target_col, edge_weight):
    """
    Builds temporal snapshots with explicit (N, seq_len, F) sequences per day.
    
    For each date d, the sequence covers [d - lag_days, d - lag_days+1, ..., d].
    This creates a true sequential view of per-node features over time.
    
    Returns a list of dicts, each containing:
      - 'date': the current date
      - 'x_seq': Tensor of shape (num_nodes, seq_len, F) - temporal node features
      - 'edge_label': Tensor of shape (num_edges,) - binary ice status for each edge
      - 'split': the train/val/test label for this date
    """
    dates = sorted(df[date_col].unique())
    num_nodes = len(node2idx)
    seq_len = lag_days + 1  # Current day + lag_days historical steps
    F = len(base_features)

    # Build quick lookup: date -> (node -> feature_vector)
    logger.info("Building per-date node feature lookup tables...")
    date_to_node_features = {}
    date_to_node_labels = {}
    date_to_node_splits = {}
    for d in dates:
        df_day = df[df[date_col] == d]
        feat_matrix = np.zeros((num_nodes, F), dtype=np.float32)
        label_vec = np.zeros(num_nodes, dtype=np.float32)
        split_vec = np.empty(num_nodes, dtype=object)
        split_vec.fill('unknown')
        for _, row in df_day.iterrows():
            idx = node2idx.get(row['node_id'])
            if idx is not None:
                feat_matrix[idx] = row[base_features].values.astype(np.float32)
                label_vec[idx] = float(row[target_col])
                split_vec[idx] = row['split']
        date_to_node_features[d] = feat_matrix
        date_to_node_labels[d] = label_vec
        date_to_node_splits[d] = split_vec

    snapshots = []
    for i, d in enumerate(dates):
        if i < lag_days:
            continue  # Skip first lag_days dates that can't form a full sequence

        # Build sequence: [d - lag_days, ..., d]
        seq_dates = [dates[i - lag_days + t] for t in range(seq_len)]
        x_seq = np.stack([date_to_node_features[sd] for sd in seq_dates], axis=1)  # (N, seq_len, F)

        # Node labels from current day
        node_labels = date_to_node_labels[d]
        node_splits = date_to_node_splits[d]
        snapshots.append({
            'date': d,
            'x_seq': torch.tensor(x_seq, dtype=torch.float32),
            'node_labels': torch.tensor(node_labels, dtype=torch.float32),
            'edge_weight': edge_weight,
            'node_splits': node_splits
        })

    return snapshots


def run_one_model(m_name, model, all_snaps, edge_index, node_areas, device, config):
    """Full training + evaluation loop for one model."""
    
    # Calculate pos_weight over only TRAIN nodes in all_snaps
    pos_total, neg_total = 0, 0
    for s in all_snaps:
        node_lbl = s['node_labels'].to(device)
        train_mask = torch.tensor(s['node_splits'] == 'train', dtype=torch.bool, device=device)
        if torch.any(train_mask):
            pos_total += node_lbl[train_mask].sum().item()
            neg_total += (1 - node_lbl[train_mask]).sum().item()
            
    pos_weight = torch.tensor([max(1.0, neg_total / (pos_total + 1e-8))], device=device)
    logger.info(f"{m_name}: pos_weight = {pos_weight.item():.2f}  (pos={int(pos_total)}, neg={int(neg_total)})")

    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, verbose=False)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_prauc = -1
    best_val_probs = None
    best_val_targets = None
    patience = config['training']['patience']
    epochs_no_improve = 0
    model_path = os.path.join(config['paths']['models_dir'], f"{m_name}_best.pt")

    epoch_logs = []

    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0

        # Shuffle snapshots each epoch
        idxs = np.random.permutation(len(all_snaps))
        num_train_batches = 0

        for idx in idxs:
            snap = all_snaps[idx]
            train_mask = torch.tensor(snap['node_splits'] == 'train', dtype=torch.bool, device=device)
            if not torch.any(train_mask):
                continue

            x = snap['x_seq'].to(device)
            node_lbl = snap['node_labels'].to(device)
            edge_weight = snap['edge_weight'].to(device)

            optimizer.zero_grad()
            logits = model(x, edge_index, edge_weight)
            
            loss = criterion(logits[train_mask], node_lbl[train_mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            num_train_batches += 1

        train_loss /= max(1, num_train_batches)

        # Validation
        model.eval()
        epoch_record = {"Epoch": epoch, "Global_Train_Loss": train_loss}
        
        # Calculate per-area losses for TRAIN_AREAS
        with torch.no_grad():
            global_val_probs, global_val_targets = [], []
            for area in TRAIN_AREAS:
                area_mask = torch.tensor(node_areas == area, dtype=torch.bool, device=device)
                
                # Area train loss
                a_tr_loss = 0.0
                tr_count = 0
                for snap in all_snaps:
                    mask = area_mask & torch.tensor(snap['node_splits'] == 'train', dtype=torch.bool, device=device)
                    if not torch.any(mask): continue
                    x = snap['x_seq'].to(device)
                    node_lbl = snap['node_labels'].to(device)
                    edge_weight = snap['edge_weight'].to(device)
                    logits = model(x, edge_index, edge_weight)
                    a_tr_loss += criterion(logits[mask], node_lbl[mask]).item()
                    tr_count += 1
                epoch_record[f"{area}_Train_Loss"] = a_tr_loss / max(1, tr_count) if tr_count > 0 else np.nan
                
                # Area val loss and metrics
                a_va_loss = 0.0
                a_va_probs, a_va_targets = [], []
                va_count = 0
                for snap in all_snaps:
                    mask = area_mask & torch.tensor(snap['node_splits'] == 'val', dtype=torch.bool, device=device)
                    if not torch.any(mask): continue
                    x = snap['x_seq'].to(device)
                    node_lbl = snap['node_labels'].to(device)
                    edge_weight = snap['edge_weight'].to(device)
                    logits = model(x, edge_index, edge_weight)
                    a_va_loss += criterion(logits[mask], node_lbl[mask]).item()
                    a_va_probs.extend(torch.sigmoid(logits[mask]).cpu().numpy())
                    a_va_targets.extend(node_lbl[mask].cpu().numpy())
                    va_count += 1
                    
                if va_count > 0:
                    a_va_metrics = compute_metrics(np.array(a_va_targets), np.array(a_va_probs))
                    epoch_record[f"{area}_Val_Loss"] = a_va_loss / va_count
                    epoch_record[f"{area}_Val_PRAUC"] = a_va_metrics["PR_AUC"]
                    epoch_record[f"{area}_Val_F1"] = a_va_metrics["F1"]
                    
                    global_val_probs.extend(a_va_probs)
                    global_val_targets.extend(a_va_targets)
                else:
                    epoch_record[f"{area}_Val_Loss"] = np.nan
                    epoch_record[f"{area}_Val_PRAUC"] = np.nan
                    epoch_record[f"{area}_Val_F1"] = np.nan

        epoch_logs.append(epoch_record)
        
        has_val = len(global_val_targets) > 0
        if has_val:
            val_metrics = compute_metrics(np.array(global_val_targets), np.array(global_val_probs))
            val_prauc = val_metrics['PR_AUC']
            not_nan = not (val_prauc != val_prauc)
            if not_nan:
                scheduler.step(val_prauc)
        
        improved = has_val and not_nan and val_prauc > best_val_prauc
        if improved or not has_val:
            if improved:
                best_val_prauc = val_prauc
                best_val_probs = list(global_val_probs)
                best_val_targets = list(global_val_targets)
                epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        elif has_val and not_nan:
            epochs_no_improve += 1

        if epoch % 20 == 0:
            prauc_str = f"{val_prauc:.4f}" if (has_val and not_nan) else "N/A"
            f1_str = f"{val_metrics['F1']:.4f}" if (has_val and not_nan) else "N/A"
            logger.info(f"[{m_name}] Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
                        f"Val PR-AUC: {prauc_str} | Val F1: {f1_str} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.5f}")

        if has_val and epochs_no_improve >= patience:
            logger.info(f"[{m_name}] Early stopping at epoch {epoch}")
            break

    # Load best and evaluate test
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Threshold: use val if available, else default 0.5
    if best_val_targets and len(best_val_targets) > 0:
        best_threshold = tune_threshold_on_validation(best_val_targets, np.array(best_val_probs), metric="F1")
    else:
        best_threshold = 0.5
        logger.warning(f"[{m_name}] No val snapshots available. Using default threshold=0.5")

    # Save loss curves
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(epoch_logs).to_excel(os.path.join(config['paths']['metrics_dir'], f"{m_name}_train_curves_{timestamp}.xlsx"), index=False)

    # Test Evaluation on 8 Graphs
    # Combine all logic so we can evaluate each test graph using ALL its snapshots (train+val+test dates).
    # Since TEST_AREAS are completely independent, we just pass all snaps and filter nodes by area mask
    all_test_metrics = []
    
    for area in TEST_AREAS:
        mask = torch.tensor(node_areas == area, dtype=torch.bool, device=device)
        if not torch.any(mask):
            continue
            
        area_probs = []
        area_targets = []
        
        with torch.no_grad():
            for snap in all_snaps:
                x = snap['x_seq'].to(device)
                node_lbl = snap['node_labels'].to(device)
                edge_weight = snap['edge_weight'].to(device)
                logits = model(x, edge_index, edge_weight)
                probs = torch.sigmoid(logits)
                area_probs.extend(probs[mask].cpu().numpy())
                area_targets.extend(node_lbl[mask].cpu().numpy())
                
        area_probs = np.array(area_probs)
        area_targets = np.array(area_targets)
        
        if len(area_targets) > 0:
            metrics = compute_metrics(area_targets, area_probs, threshold=best_threshold)
            metrics['Model'] = m_name
            metrics['Area'] = area
            metrics['Threshold'] = best_threshold
            all_test_metrics.append(metrics)
            
    df_preds = pd.DataFrame(all_test_metrics)
    df_preds.to_excel(os.path.join(config['paths']['metrics_dir'], f"{m_name}_test_metrics_{timestamp}.xlsx"), index=False)
    
    logger.info(f"[{m_name}] Evaluated on test areas and saved metrics.")
    
    # Calculate global test metrics just for logging
    if len(all_test_metrics) > 0:
        avg_prauc = df_preds['PR_AUC'].mean()
        avg_f1 = df_preds['F1'].mean()
        logger.info(f"[{m_name}] Test Average PR-AUC: {avg_prauc:.4f} | Average F1: {avg_f1:.4f}")
        return all_test_metrics[0] # returning one just for aggregation
    return None


def train_gnn():
    config = load_config()
    set_seed(config['project']['seed'])

    data_path = os.path.join(config['paths']['processed_data_dir'], "processed_tabular.csv")
    df = pd.read_csv(data_path)
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

    with open(os.path.join(config['paths']['processed_data_dir'], "schema_report.json"), "r") as f:
        schema = json.load(f)

    target = schema['target_col']
    all_features = schema['all_model_features']

    # Base per-time-step features for the Temporal GNN's recurrent input
    gnn_cfg = config['gnn']
    base_features = [f for f in gnn_cfg['temporal_base_features'] if f in df.columns]
    seq_len = gnn_cfg['seq_len']
    lag_days = seq_len - 1  # e.g. seq_len=4 → lag_days=3

    logger.info(f"Temporal GNN base features: {len(base_features)} | seq_len: {seq_len}")
    logger.info(f"Static GNN total features: {len(all_features)}")

    # ---------------------------------------------------------------
    # Standardize
    # ---------------------------------------------------------------
    # Fit scaler only on train rows AND train areas
    train_mask = (df['split'] == 'train') & (df['area_id'].isin(TRAIN_AREAS))
    scaler_all = StandardScaler()
    scaler_all.fit(df.loc[train_mask, all_features])
    df_scaled = df.copy()
    df_scaled[all_features] = scaler_all.transform(df[all_features])

    # Separate scaler for base_features (used by TemporalGNN's explicit sequence)
    scaler_base = StandardScaler()
    scaler_base.fit(df.loc[train_mask, base_features])
    df_scaled[base_features] = scaler_base.transform(df[base_features])  # note: overlaps with all_features but OK

    with open(os.path.join(config['paths']['models_dir'], "gnn_scaler_all.pkl"), "wb") as f:
        pickle.dump(scaler_all, f)
    with open(os.path.join(config['paths']['models_dir'], "gnn_scaler_base.pkl"), "wb") as f:
        pickle.dump(scaler_base, f)

    # ---------------------------------------------------------------
    # Build static graph
    # ---------------------------------------------------------------
    node2idx, edge_index, edge_weight = build_static_graph_structure(
        df_scaled, entity_col=config['data']['entity_col'],
        edge_list_col=config['data']['edge_list_col']
    )
    logger.info(f"Static graph: {len(node2idx)} nodes, {edge_index.size(1)} undirected edges")

    if edge_index.shape[1] == 0:
        logger.error("No edges found! Check edge_list_col in config.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = edge_index.to(device)

    # Pre-compute node areas
    idx2node = {v: k for k, v in node2idx.items()}
    node_areas = np.array([df[df['node_id'] == idx2node[i]]['area_id'].values[0] for i in range(len(node2idx))])

    # ---------------------------------------------------------------
    # Build temporal snapshots for TemporalNodeGNN
    # ---------------------------------------------------------------
    logger.info("Building explicit temporal sequence snapshots (for TemporalNodeGNN)...")
    temporal_snaps = build_temporal_graph_sequence(
        df_scaled, node2idx, base_features, lag_days,
        date_col=config['data']['date_col'], target_col=target, edge_weight=edge_weight
    )
    logger.info(f"TemporalGNN snapshots -> Total sequences: {len(temporal_snaps)}")

    # ---------------------------------------------------------------
    # Build flat snapshots for StaticNodeGNN
    # ---------------------------------------------------------------
    logger.info("Generating flattened snapshots for StaticNodeGNN...")
    static_snaps = []
    num_nodes = len(node2idx)
    for d in dates:
        df_d = df_scaled[df_scaled['date'] == d]
        x = np.zeros((num_nodes, len(all_features)), dtype=np.float32)
        labels = np.zeros(num_nodes, dtype=np.float32)
        split_vec = np.empty(num_nodes, dtype=object)
        split_vec.fill('unknown')
        for _, row in df_d.iterrows():
            idx = node2idx.get(row['node_id'])
            if idx is not None:
                x[idx] = row[all_features].values.astype(np.float32)
                labels[idx] = float(row[target])
                split_vec[idx] = row['split']
        static_snaps.append({
            'date': d,
            'x_seq': torch.tensor(x, dtype=torch.float32),  # (N, F) - flat
            'node_labels': torch.tensor(labels, dtype=torch.float32),
            'edge_weight': edge_weight,
            'node_splits': split_vec
        })

    # ---------------------------------------------------------------
    # Models to compare
    # ---------------------------------------------------------------
    hidden_dim = gnn_cfg['hidden_dim']
    dropout = gnn_cfg['dropout']
    temporal_rnn = gnn_cfg['temporal_model']
    models_to_run = {
        "TemporalGNN_GCN_GRU": (
            TemporalNodeGNN(
                input_dim=len(base_features), hidden_dim=hidden_dim,
                seq_len=gnn_cfg['seq_len'], dropout=dropout,
                gnn_type='GCN', temporal_model='gru'
            ).to(device),
            temporal_snaps
        ),
        "TemporalGNN_GCN_LSTM": (
            TemporalNodeGNN(
                input_dim=len(base_features), hidden_dim=hidden_dim,
                seq_len=gnn_cfg['seq_len'], dropout=dropout,
                gnn_type='GCN', temporal_model='lstm'
            ).to(device),
            temporal_snaps
        ),
        "TemporalGNN_SAGE_GRU": (
            TemporalNodeGNN(
                input_dim=len(base_features), hidden_dim=hidden_dim,
                seq_len=gnn_cfg['seq_len'], dropout=dropout,
                gnn_type='SAGE', temporal_model='gru'
            ).to(device),
            temporal_snaps
        ),
        "TemporalGNN_SAGE_LSTM": (
            TemporalNodeGNN(
                input_dim=len(base_features), hidden_dim=hidden_dim,
                seq_len=gnn_cfg['seq_len'], dropout=dropout,
                gnn_type='SAGE', temporal_model='lstm'
            ).to(device),
            temporal_snaps
        ),
        "TemporalGNN_GAT_GRU": (
            TemporalNodeGNN(
                input_dim=len(base_features), hidden_dim=hidden_dim,
                seq_len=gnn_cfg['seq_len'], dropout=dropout,
                gnn_type='GAT', temporal_model='gru'
            ).to(device),
            temporal_snaps
        ),
        "TemporalGNN_GAT_LSTM": (
            TemporalNodeGNN(
                input_dim=len(base_features), hidden_dim=hidden_dim,
                seq_len=gnn_cfg['seq_len'], dropout=dropout,
                gnn_type='GAT', temporal_model='lstm'
            ).to(device),
            temporal_snaps
        ),
        "StaticGNN_ablation": (
            StaticNodeGNN(
                input_dim=len(all_features), hidden_dim=hidden_dim,
                dropout=dropout, gnn_type='GCN'
            ).to(device),
            static_snaps
        ),
    }

    all_metrics = []
    for m_name, (model, all_snaps) in models_to_run.items():
        logger.info(f"\n{'='*60}\nTraining {m_name}...\n{'='*60}")
        metrics = run_one_model(m_name, model, all_snaps, edge_index, node_areas, device, config)
        if metrics:
            all_metrics.append(metrics)

    if len(all_metrics) > 0:
        df_metrics = pd.DataFrame(all_metrics)
        metrics_out = os.path.join(config['paths']['metrics_dir'], "gnn_metrics_summary.csv")
        df_metrics.to_csv(metrics_out, index=False)
        logger.info(f"\nGNN metrics summarized to {metrics_out}")


if __name__ == "__main__":
    train_gnn()
