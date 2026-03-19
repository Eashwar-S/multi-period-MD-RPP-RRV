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
from models.temporal_gnn import TemporalEdgeGNN, StaticEdgeGNN
from utils.reproducibility import set_seed
from sklearn.preprocessing import StandardScaler

logger = setup_logger("TrainGNN", "outputs/logs/train_gnn.log")


def build_temporal_graph_sequence(df, node2idx, base_features, lag_days, date_col, target_col):
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
    for d in dates:
        df_day = df[df[date_col] == d]
        feat_matrix = np.zeros((num_nodes, F), dtype=np.float32)
        label_vec = np.zeros(num_nodes, dtype=np.float32)
        for _, row in df_day.iterrows():
            idx = node2idx.get(row['node_id'])
            if idx is not None:
                feat_matrix[idx] = row[base_features].values.astype(np.float32)
                label_vec[idx] = float(row[target_col])
        date_to_node_features[d] = feat_matrix
        date_to_node_labels[d] = label_vec

    # Get split mapping
    date_to_split = df[[date_col, 'split']].drop_duplicates().set_index(date_col)['split'].to_dict()

    snapshots = []
    for i, d in enumerate(dates):
        if i < lag_days:
            continue  # Skip first lag_days dates that can't form a full sequence

        # Build sequence: [d - lag_days, ..., d]
        seq_dates = [dates[i - lag_days + t] for t in range(seq_len)]
        x_seq = np.stack([date_to_node_features[sd] for sd in seq_dates], axis=1)  # (N, seq_len, F)

        # Edge labels from current day: edge is icy if either endpoint node is icy
        node_labels = date_to_node_labels[d]
        snapshots.append({
            'date': d,
            'x_seq': torch.tensor(x_seq, dtype=torch.float32),
            'node_labels': torch.tensor(node_labels, dtype=torch.float32),
            'split': date_to_split.get(d, 'unknown')
        })

    return snapshots


def compute_edge_labels(node_labels_t, edge_index):
    """Edge is icy if either incident node is icy (OR logic)."""
    u, v = edge_index[0], edge_index[1]
    return torch.max(node_labels_t[u], node_labels_t[v])


def run_one_model(m_name, model, train_snaps, val_snaps, test_snaps, edge_index, device, config):
    """Full training + evaluation loop for one model."""
    pos_total = sum(compute_edge_labels(s['node_labels'].to(device), edge_index).sum().item() for s in train_snaps)
    neg_total = sum((1 - compute_edge_labels(s['node_labels'].to(device), edge_index)).sum().item() for s in train_snaps)
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

    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0

        # Shuffle training snapshots each epoch
        idxs = np.random.permutation(len(train_snaps))

        for idx in idxs:
            snap = train_snaps[idx]
            x = snap['x_seq'].to(device)            # (N, seq_len, F)  or (N, F) for static
            edge_lbl = compute_edge_labels(snap['node_labels'].to(device), edge_index)

            optimizer.zero_grad()
            logits = model(x, edge_index)
            loss = criterion(logits, edge_lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_snaps)

        # Validation (guard against empty val set)
        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for snap in val_snaps:
                x = snap['x_seq'].to(device)
                edge_lbl = compute_edge_labels(snap['node_labels'].to(device), edge_index)
                logits = model(x, edge_index)
                probs = torch.sigmoid(logits)
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(edge_lbl.cpu().numpy())

        val_metrics = compute_metrics(val_targets, np.array(val_probs))
        val_prauc = val_metrics['PR_AUC']
        
        # Only use scheduler and early stopping when we have real val metrics
        has_val = len(val_targets) > 0 and not (val_prauc != val_prauc)  # NaN check
        if has_val:
            scheduler.step(val_prauc)
        
        improved = has_val and val_prauc > best_val_prauc
        if improved or not has_val:
            if improved:
                best_val_prauc = val_prauc
                best_val_probs = list(val_probs)
                best_val_targets = list(val_targets)
                epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        elif has_val:
            epochs_no_improve += 1

        if epoch % 20 == 0:
            prauc_str = f"{val_prauc:.4f}" if has_val else "N/A"
            f1_str = f"{val_metrics['F1']:.4f}" if has_val else "N/A"
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

    test_probs, test_targets = [], []
    with torch.no_grad():
        for snap in test_snaps:
            x = snap['x_seq'].to(device)
            edge_lbl = compute_edge_labels(snap['node_labels'].to(device), edge_index)
            logits = model(x, edge_index)
            probs = torch.sigmoid(logits)
            test_probs.extend(probs.cpu().numpy())
            test_targets.extend(edge_lbl.cpu().numpy())

    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)

    test_metrics = compute_metrics(test_targets, test_probs, threshold=best_threshold)
    test_metrics['Model'] = m_name
    test_metrics['Threshold'] = best_threshold
    logger.info(f"[{m_name}] Test PR-AUC: {test_metrics['PR_AUC']:.4f} | F1: {test_metrics['F1']:.4f} "
                f"| Recall: {test_metrics['Recall']:.4f} | Precision: {test_metrics['Precision']:.4f}")

    plot_pr_curve(test_targets, test_probs, m_name,
                  os.path.join(config['paths']['plots_dir'], f"pr_curve_{m_name}.png"))
    plot_confusion_matrix(test_targets, (test_probs >= best_threshold).astype(int), m_name,
                          os.path.join(config['paths']['plots_dir'], f"confusion_matrix_{m_name}.png"))

    df_preds = pd.DataFrame({
        "true": test_targets, "prob": test_probs,
        "pred": (test_probs >= best_threshold).astype(int)
    })
    df_preds.to_csv(os.path.join(config['paths']['predictions_dir'], f"{m_name}_test_predictions.csv"), index=False)

    return test_metrics


def train_gnn():
    config = load_config()
    set_seed(config['project']['seed'])

    data_path = os.path.join(config['paths']['processed_data_dir'], "processed_tabular.csv")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)

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
    # Fit scaler only on train rows for all_features (used by StaticGNN)
    train_mask = df['split'] == 'train'
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
    node2idx, edge_index = build_static_graph_structure(
        df_scaled, entity_col=config['data']['entity_col'],
        edge_list_col=config['data']['edge_list_col']
    )
    logger.info(f"Static graph: {len(node2idx)} nodes, {edge_index.shape[1]//2} undirected edges")

    if edge_index.shape[1] == 0:
        logger.error("No edges found! Check edge_list_col in config.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = edge_index.to(device)

    # ---------------------------------------------------------------
    # Build temporal snapshots for TemporalEdgeGNN
    # ---------------------------------------------------------------
    logger.info("Building explicit temporal sequence snapshots (for TemporalEdgeGNN)...")
    temporal_snaps = build_temporal_graph_sequence(
        df_scaled, node2idx, base_features, lag_days,
        date_col=config['data']['date_col'], target_col=target
    )
    train_t = [s for s in temporal_snaps if s['split'] == 'train']
    val_t = [s for s in temporal_snaps if s['split'] == 'val']
    test_t = [s for s in temporal_snaps if s['split'] == 'test']
    logger.info(f"TemporalGNN snapshots -> Train: {len(train_t)}, Val: {len(val_t)}, Test: {len(test_t)}")

    # ---------------------------------------------------------------
    # Build flat snapshots for StaticEdgeGNN
    # ---------------------------------------------------------------
    dates = sorted(df_scaled['date'].unique())
    date_to_split = df_scaled[['date', 'split']].drop_duplicates().set_index('date')['split'].to_dict()
    num_nodes = len(node2idx)

    static_snaps = []
    for d in dates:
        df_d = df_scaled[df_scaled['date'] == d]
        x = np.zeros((num_nodes, len(all_features)), dtype=np.float32)
        labels = np.zeros(num_nodes, dtype=np.float32)
        for _, row in df_d.iterrows():
            idx = node2idx.get(row['node_id'])
            if idx is not None:
                x[idx] = row[all_features].values.astype(np.float32)
                labels[idx] = float(row[target])
        static_snaps.append({
            'date': d,
            'x_seq': torch.tensor(x, dtype=torch.float32),  # (N, F) - flat
            'node_labels': torch.tensor(labels, dtype=torch.float32),
            'split': date_to_split.get(d, 'unknown')
        })

    train_s = [s for s in static_snaps if s['split'] == 'train']
    val_s = [s for s in static_snaps if s['split'] == 'val']
    test_s = [s for s in static_snaps if s['split'] == 'test']

    # ---------------------------------------------------------------
    # Models to compare
    # ---------------------------------------------------------------
    hidden_dim = gnn_cfg['hidden_dim']
    dropout = gnn_cfg['dropout']
    temporal_rnn = gnn_cfg['temporal_model']

    models_to_run = {
        "TemporalGNN_GCN": (
            TemporalEdgeGNN(
                input_dim=len(base_features), hidden_dim=hidden_dim,
                seq_len=seq_len, dropout=dropout, gnn_type='GCN',
                temporal_model=temporal_rnn
            ).to(device),
            train_t, val_t, test_t
        ),
        "TemporalGNN_SAGE": (
            TemporalEdgeGNN(
                input_dim=len(base_features), hidden_dim=hidden_dim,
                seq_len=seq_len, dropout=dropout, gnn_type='SAGE',
                temporal_model=temporal_rnn
            ).to(device),
            train_t, val_t, test_t
        ),
        "TemporalGNN_GAT": (
            TemporalEdgeGNN(
                input_dim=len(base_features), hidden_dim=hidden_dim,
                seq_len=seq_len, dropout=dropout, gnn_type='GAT',
                temporal_model=temporal_rnn
            ).to(device),
            train_t, val_t, test_t
        ),
        "StaticGNN_ablation": (
            StaticEdgeGNN(
                input_dim=len(all_features), hidden_dim=hidden_dim,
                dropout=dropout, gnn_type='GCN'
            ).to(device),
            train_s, val_s, test_s
        ),
    }

    all_metrics = []
    for m_name, (model, tr, vl, te) in models_to_run.items():
        logger.info(f"\n{'='*60}\nTraining {m_name}...\n{'='*60}")
        metrics = run_one_model(m_name, model, tr, vl, te, edge_index, device, config)
        all_metrics.append(metrics)

    df_metrics = pd.DataFrame(all_metrics)
    metrics_out = os.path.join(config['paths']['metrics_dir'], "gnn_metrics.csv")
    df_metrics.to_csv(metrics_out, index=False)
    logger.info(f"\nGNN metrics saved to {metrics_out}")
    logger.info(df_metrics[['Model', 'PR_AUC', 'F1', 'Recall', 'Precision']].to_string(index=False))


if __name__ == "__main__":
    train_gnn()
