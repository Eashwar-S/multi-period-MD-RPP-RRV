#!/usr/bin/env python3
"""
generate_gnn_routing_instances.py  (v2 – full overhaul)
=========================================================
Runs ONCE to build reusable routing instances for the 8 GNN test cities.

Changes from v1
---------------
* Per-city bounding-box and OSM tolerance loaded from Final_dataset_info.xlsx
  (hardcoded below to avoid runtime Excel dependency).
* Weather features extracted DIRECTLY from raw graph_data/ CSVs for each city,
  then processed through the same create_temporal_features pipeline used at
  training time, and scaled with the saved GNN scaler.
* OSM graph is filtered to its largest weakly-connected component before use.
* Depot placement uses a GREEDY SET-COVER algorithm:
  every directed edge (u,v) must be reachable from at least one depot with a
  round-trip cost ≤ FLIGHT_BUDGET_S (depot→u + w(u,v) + v→depot).
* recharge_time = 1.5 × FLIGHT_BUDGET_S.

Environment: run from multi_period_routing/ or any directory.
"""

import os
import sys
import json
import pickle
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import networkx as nx
from scipy.spatial import cKDTree

try:
    import osmnx as ox
except ImportError:
    sys.exit("osmnx is required: pip install osmnx")

# ── Project-local imports ──────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.io_utils import load_config
from utils.graph_utils import build_static_graph_structure
from utils.feature_utils import create_temporal_features
from utils.reproducibility import set_seed
from models.temporal_gnn import TemporalNodeGNN
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('GenRouting')

# ══════════════════════════════════════════════════════════════════════════════
# Per-city configuration (from Final_dataset_info.xlsx)
# East longitude for Little Rock corrected from 92.24 → -92.24
# ══════════════════════════════════════════════════════════════════════════════
AREA_CONFIG = {
    'Little Rock, Arkansas, US': {
        'tolerance': 125,
        'north': 34.80,  'east': -92.24,
        'south': 34.70,  'west': -92.34,
        'csv': 'Little Rock, Arkansas, US_Jan_2026.csv',
    },
    'Louisville, Kentucky, US': {
        'tolerance': 115,
        'north': 38.30,  'east': -85.71,
        'south': 38.20,  'west': -85.81,
        'csv': 'Louisville, Kentucky, US_Jan_2026.csv',
    },
    'Lubbock, Texaas, US': {          # ← "Texaas" typo matches training code
        'tolerance': 160,
        'north': 33.6261, 'east': -101.7918,
        'south': 33.5237, 'west': -101.9435,
        'csv': 'Lubbock, Texaas, US_Jan_2026.csv',
    },
    'Memphis, US': {
        'tolerance': 130,
        'north': 35.20,  'east': -90.00,
        'south': 35.10,  'west': -90.10,
        'csv': 'Memphis, US_Jan_2026.csv',
    },
    'Nashville, Tennessee, US': {
        'tolerance': 135,
        'north': 36.21,  'east': -86.73,
        'south': 36.11,  'west': -86.83,
        'csv': 'Nashville, Tennessee, US_Jan_2026.csv',
    },
    'newyork, US': {
        'tolerance': 130,
        'north': 40.75,  'east': -73.90,
        'south': 40.65,  'west': -74.00,
        'csv': 'newyork, US_Jan_2026.csv',
    },
    'Oklahoma, US': {
        'tolerance': 135,
        'north': 35.52,  'east': -97.47,
        'south': 35.42,  'west': -97.58,
        'csv': 'Oklahoma, US_Jan_2026.csv',
    },
    'Philadelphia, US': {
        'tolerance': 130,
        'north': 39.99619, 'east': -75.12794,
        'south': 39.89460, 'west': -75.29016,
        'csv': 'Philadelphia, US_Jan_2026.csv',
    },
}

# ── Physical constants ─────────────────────────────────────────────────────────
DRONE_SPEED_MS   = 20.0            # metres / second
FLIGHT_BUDGET_S  = 40 * 60        # 40-min battery (seconds)
RECHARGE_TIME_S  = 1.5 * FLIGHT_BUDGET_S   # = 3600 s


# ══════════════════════════════════════════════════════════════════════════════
# §1  OSM download helpers
# ══════════════════════════════════════════════════════════════════════════════

def download_osm_graph(area, cfg):
    """Download, consolidate, simplify OSM driving graph using exact bbox from config."""
    roads_filter = '["highway"~"motorway|trunk|primary|secondary"]'
    n, e, s, w = cfg['north'], cfg['east'], cfg['south'], cfg['west']
    tol = cfg['tolerance']
    log.info(f"  Downloading OSM (tol={tol}m)  N={n} E={e} S={s} W={w}")

    G = ox.graph_from_bbox(
        bbox=(w, s, e, n),
        retain_all=False,
        custom_filter=roads_filter,
        network_type='drive',
    )
    G_proj = ox.project_graph(G.copy())
    G = ox.consolidate_intersections(G_proj, tolerance=tol,
                                      rebuild_graph=True, dead_ends=True)
    G_di = ox.convert.to_digraph(G, weight='length')
    G = nx.MultiDiGraph(G_di)
    if "simplified" in G.graph:
        del G.graph["simplified"]
    for u, v, key, data in G.edges(keys=True, data=True):
        for attr_key, attr_val in list(data.items()):
            if isinstance(attr_val, list):
                data[attr_key] = str(attr_val)
    G = ox.simplify_graph(G, edge_attrs_differ=[])
    G = ox.project_graph(G, to_crs="EPSG:4326")
    log.info(f"  Raw OSM MultiDiGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def to_simple_digraph(G_multi):
    """
    Collapse MultiDiGraph → simple DiGraph keeping minimum-weight directed edge
    per (u, v) pair.  The SA solver requires G[u][v]['weight'] direct access.
    """
    G_s = nx.DiGraph()
    for node, data in G_multi.nodes(data=True):
        G_s.add_node(node, **data)
    for u, v, data in G_multi.edges(data=True):
        w = data.get('weight', float('inf'))
        if G_s.has_edge(u, v):
            if w < G_s[u][v].get('weight', float('inf')):
                G_s[u][v].update(data)
        else:
            G_s.add_edge(u, v, **data)
    return G_s


def set_time_weights(G):
    """
    Replace 'length' (metres) with 'weight' (seconds = length / DRONE_SPEED_MS).
    Works on both MultiDiGraph and DiGraph.
    """
    edges = (G.edges(keys=True, data=True)
             if isinstance(G, nx.MultiDiGraph) else G.edges(data=True))
    for item in edges:
        data = item[-1]
        length_m = data.get('length', 1.0)
        if isinstance(length_m, str):
            try:
                length_m = float(length_m)
            except ValueError:
                length_m = 1.0
        data['length_m'] = length_m
        data['weight']   = length_m / DRONE_SPEED_MS
    return G


def filter_to_largest_wcc(G):
    """
    Keep only the largest weakly-connected component of G.
    Returns a new DiGraph restricted to that component.
    """
    wcc = max(nx.weakly_connected_components(G), key=len)
    G_sub = G.subgraph(wcc).copy()
    log.info(f"  Largest WCC: {G_sub.number_of_nodes()} nodes, "
             f"{G_sub.number_of_edges()} edges "
             f"(removed {G.number_of_nodes()-G_sub.number_of_nodes()} nodes)")
    return G_sub


# ══════════════════════════════════════════════════════════════════════════════
# §2  Coverage-aware depot placement
# ══════════════════════════════════════════════════════════════════════════════

def coverage_aware_depots(G, dist, capacity):
    """
    Greedy set-cover: find a minimal set of depots so that EVERY directed edge
    (u, v) is reachable from at least one depot with a round-trip cost ≤ capacity:
        dist[depot][u]  +  w(u,v)  +  dist[v][depot]  ≤ capacity

    Parameters
    ----------
    G        : nx.DiGraph with 'weight' edge attribute
    dist     : all-pairs shortest-path dict (node → {node: distance})
    capacity : float – maximum single-flight budget (seconds)

    Returns
    -------
    depots : list of node IDs
    """
    all_directed_edges = list(G.edges())
    all_nodes = list(G.nodes())

    log.info(f"  Computing coverage sets for {len(all_nodes)} candidate depots "
             f"over {len(all_directed_edges)} directed edges ...")

    # For each candidate depot, which directed edges can it cover?
    node_covers: dict = {}
    for d in all_nodes:
        d_dist = dist.get(d, {})
        covered = set()
        for u, v in all_directed_edges:
            w   = G[u][v].get('weight', float('inf'))
            dtu = d_dist.get(u, float('inf'))
            vtd = dist.get(v, {}).get(d, float('inf'))
            if dtu + w + vtd <= capacity + 1e-6:
                covered.add((u, v))
        node_covers[d] = covered

    # Greedy set-cover
    uncovered = set(all_directed_edges)
    depots    = []
    used      = set()

    while uncovered:
        # Pick the un-used node covering the most uncovered edges
        best = max(
            (n for n in all_nodes if n not in used),
            key=lambda n: len(node_covers[n] & uncovered),
            default=None,
        )
        if best is None or not (node_covers[best] & uncovered):
            break   # remaining edges are truly unreachable
        depots.append(best)
        used.add(best)
        uncovered -= node_covers[best]

    n_uncoverable = len(uncovered)
    log.info(f"  Depots selected: {len(depots)} | "
             f"Uncoverable edges (truly unreachable): {n_uncoverable}")
    return depots


# ══════════════════════════════════════════════════════════════════════════════
# §3  Weather / feature pipeline for a single test city
# ══════════════════════════════════════════════════════════════════════════════

def load_area_features(area, csv_filename, config, global_node2idx, scaler_base, base_features):
    """
    Load the raw CSV for `area` from graph_data/, apply the same temporal
    feature engineering used during training, scale with `scaler_base`, and
    return a DataFrame whose index is date and columns are base_features —
    ready for building GNN snapshots.

    Returns
    -------
    df_area_scaled : pd.DataFrame  (with 'node_id', 'date', base_features, ...)
    valid_node_ids : list of node_ids in this area that exist in global_node2idx
    """
    csv_path = os.path.join(PROJECT_ROOT, config['paths']['raw_data_dir'], csv_filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['area_id'] = area

    # Fill NaNs (same as build_temporal_dataset.py)
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Create temporal lag / rolling features
    # Include 'icy_label' so icy_label_lag1 (the 22nd GNN base feature) is created
    weather_cols = config['data']['weather_cols']
    cols_to_lag  = (weather_cols
                    + ['icy_base', 'severity', 'pavement_temp', 'ice_thickness', 'icy_label'])
    cols_to_lag  = [c for c in cols_to_lag if c in df.columns]

    df_feat = create_temporal_features(
        df,
        entity_col  = config['data']['entity_col'],
        date_col    = config['data']['date_col'],
        cols_to_lag = cols_to_lag,
        target_col  = config['data']['target_col'],
        max_lag     = config['data']['lag_days'],
        window_size = config['data']['rolling_window'],
    )
    df_feat = df_feat.dropna().reset_index(drop=True)

    # Ensure ALL base_features are present — zero-fill any absent columns
    # so the scaler always receives exactly n_features_in_ = 22 columns.
    for bf in base_features:
        if bf not in df_feat.columns:
            df_feat[bf] = 0.0
            log.warning(f"  [{area}] Feature '{bf}' absent after engineering — zero-filled.")

    # Scale with training scaler (transform, NOT fit)
    df_scaled = df_feat.copy()
    X_raw = df_feat[base_features].values.astype(np.float32)
    df_scaled[base_features] = scaler_base.transform(X_raw)

    # Report which features were actually present before zero-filling
    avail_bf = base_features  # always use full list (zero-filled if needed)

    # Filter to nodes in the global GNN graph
    valid_node_ids = [nid for nid in df_scaled['node_id'].unique()
                      if nid in global_node2idx]
    log.info(f"  [{area}] {len(valid_node_ids)} nodes found in global GNN graph "
             f"(out of {df_scaled['node_id'].nunique()} in CSV)")

    return df_scaled, avail_bf, valid_node_ids


def build_full_graph_snapshots(df_area_scaled, avail_bf, valid_node_ids,
                               global_node2idx, N_global, seq_len, lag_days):
    """
    Build a list of (N_global, seq_len, F) tensors — one per valid date —
    with the area's nodes filled in and all others zeroed.

    Returns
    -------
    valid_dates : list of Timestamps
    snapshots   : list of torch.Tensor (N_global, seq_len, F)
    """
    F = len(avail_bf)
    dates = sorted(df_area_scaled['date'].unique())

    # Local index map: node_id → contiguous 0-based index (area-local)
    local_node_ids = sorted(valid_node_ids, key=lambda x: global_node2idx[x])
    local_idx_map  = {nid: i for i, nid in enumerate(local_node_ids)}
    local2global   = {i: global_node2idx[nid] for i, nid in enumerate(local_node_ids)}
    N_area = len(local_node_ids)

    # Per-date area feature matrices
    date_feats: dict = {}
    for d in dates:
        df_d = df_area_scaled[df_area_scaled['date'] == d]
        mat  = np.zeros((N_area, F), dtype=np.float32)
        for _, row in df_d.iterrows():
            li = local_idx_map.get(row['node_id'])
            if li is not None:
                vals = row[avail_bf].values.astype(np.float32)
                # Protect against shape mismatch (missing features → zeros)
                mat[li, :len(vals)] = vals
        date_feats[d] = mat

    snapshots   = []
    valid_dates = []
    for i, d in enumerate(dates):
        if i < lag_days:
            continue
        seq_dates = [dates[i - lag_days + t] for t in range(seq_len)]
        x_area = np.stack([date_feats[sd] for sd in seq_dates], axis=1)  # (N_area, seq_len, F)
        x_full = np.zeros((N_global, seq_len, F), dtype=np.float32)
        for li, gi in local2global.items():
            x_full[gi] = x_area[li]
        snapshots.append(torch.tensor(x_full, dtype=torch.float32))
        valid_dates.append(d)

    return valid_dates, snapshots


# ══════════════════════════════════════════════════════════════════════════════
# §4  GNN inference
# ══════════════════════════════════════════════════════════════════════════════

def infer_node_probs(model, snapshots, edge_index, edge_weight, device):
    """Return list of np.array(N_global,) — one per snapshot/day."""
    all_probs = []
    with torch.no_grad():
        for x_full in snapshots:
            logits = model(x_full.to(device), edge_index.to(device), edge_weight.to(device))
            probs  = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_probs.append(probs)
    return all_probs


# ══════════════════════════════════════════════════════════════════════════════
# §5  Map GNN node probs → OSM edge probs
# ══════════════════════════════════════════════════════════════════════════════

def map_probs_to_edges(G_osm, gnn_node_coords, all_day_probs):
    """
    For each undirected canonical edge (min_node, max_node) in the OSM graph,
    find the nearest GNN training nodes to each endpoint and average their probs.

    Returns
    -------
    edge_probs_all : dict{ (u, v) → list[float] }  – one float per day
    """
    kdtree = cKDTree(gnn_node_coords)

    osm_nodes  = list(G_osm.nodes())
    osm_coords = np.array([[G_osm.nodes[n].get('y', 0.0),
                            G_osm.nodes[n].get('x', 0.0)] for n in osm_nodes])
    _, nn_idx  = kdtree.query(osm_coords)
    node2gnn   = {n: nn_idx[i] for i, n in enumerate(osm_nodes)}

    edge_probs_all: dict = {}
    seen: set = set()
    for u, v in G_osm.edges():
        canon = (min(u, v), max(u, v))
        if canon in seen:
            continue
        seen.add(canon)
        gi_u = node2gnn[u]
        gi_v = node2gnn[v]
        edge_probs_all[canon] = [
            float((day[gi_u] + day[gi_v]) / 2.0) for day in all_day_probs
        ]

    return edge_probs_all


# ══════════════════════════════════════════════════════════════════════════════
# §6  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def area_slug(name):
    return name.replace(', ', '_').replace(' ', '_')


def main():
    config = load_config(os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml'))
    set_seed(config['project']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    out_root = os.path.join(SCRIPT_DIR, 'gnn_routing_instances')
    os.makedirs(out_root, exist_ok=True)

    # ── GNN feature config ────────────────────────────────────────────────────
    gnn_cfg      = config['gnn']
    seq_len      = gnn_cfg['seq_len']
    lag_days     = seq_len - 1
    with open(os.path.join(PROJECT_ROOT, config['paths']['processed_data_dir'],
                           'schema_report.json')) as f:
        schema = json.load(f)
    # Base features: use only those actually present after training
    base_features_cfg = gnn_cfg['temporal_base_features']

    # ── Load global graph structure (node2idx, edge_index, edge_weight) ────────
    # We still need the global graph for the GNN's edge_index.
    # Load processed_tabular.csv ONLY to rebuild node2idx / edge_index.
    log.info("Loading global GNN graph structure ...")
    data_path = os.path.join(PROJECT_ROOT, config['paths']['processed_data_dir'],
                             'processed_tabular.csv')
    df_global = pd.read_csv(data_path, usecols=['node_id', 'edge_list',
                                                  'latitude', 'longitude'])
    node2idx, edge_index, edge_weight = build_static_graph_structure(
        df_global,
        entity_col   = config['data']['entity_col'],
        edge_list_col= config['data']['edge_list_col'],
    )
    N_global = len(node2idx)
    log.info(f"Global GNN graph: {N_global} nodes, {edge_index.size(1)} edges")

    # Global GNN node coordinates for nearest-neighbour matching
    node_snapshot = df_global.drop_duplicates(subset=['node_id']).set_index('node_id')
    gnn_node_coords = np.zeros((N_global, 2), dtype=np.float64)
    for nid, idx in node2idx.items():
        if nid in node_snapshot.index:
            row = node_snapshot.loc[nid]
            gnn_node_coords[idx] = [row.get('latitude', 0.0), row.get('longitude', 0.0)]

    # ── Load scalers ──────────────────────────────────────────────────────────
    scaler_base_path = os.path.join(PROJECT_ROOT, config['paths']['models_dir'],
                                    'gnn_scaler_base.pkl')
    with open(scaler_base_path, 'rb') as f:
        scaler_base: StandardScaler = pickle.load(f)

    # Determine actual base_features (those the scaler was fitted on)
    # The scaler was fitted with exactly len(base_features_cfg) features
    # We'll filter to those that are in the scaler's feature set via count
    n_scaler_feats = scaler_base.n_features_in_
    base_features  = base_features_cfg[:n_scaler_feats]  # trimmed to scaler width

    # ── Load GNN model ────────────────────────────────────────────────────────
    model_path = os.path.join(PROJECT_ROOT, config['paths']['models_dir'],
                              'TemporalGNN_GCN_LSTM_best.pt')
    model = TemporalNodeGNN(
        input_dim    = len(base_features),
        hidden_dim   = gnn_cfg['hidden_dim'],
        seq_len      = seq_len,
        dropout      = gnn_cfg['dropout'],
        gnn_type     = 'GCN',
        temporal_model = 'lstm',
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    log.info(f"GNN model loaded from {model_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Per-area processing
    # ══════════════════════════════════════════════════════════════════════════
    for area, cfg in AREA_CONFIG.items():
        slug      = area_slug(area)
        area_out  = os.path.join(out_root, slug)
        os.makedirs(area_out, exist_ok=True)

        graph_pkl = os.path.join(area_out, 'graph.pickle')
        edge_npy  = os.path.join(area_out, 'edge_probs.npy')
        meta_npy  = os.path.join(area_out, 'meta.npy')

        if all(os.path.exists(p) for p in [graph_pkl, edge_npy, meta_npy]):
            log.info(f"[SKIP] {area} – instance already exists.")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"Processing: {area}")
        log.info(f"{'='*60}")

        # ── 1. Download OSM graph ─────────────────────────────────────────────
        try:
            G_osm = download_osm_graph(area, cfg)
        except Exception as e:
            log.error(f"OSM download failed for {area}: {e}")
            continue

        # ── 2. Convert edge weights metres → seconds ──────────────────────────
        G_osm = set_time_weights(G_osm)

        # ── 3. Convert to simple DiGraph ──────────────────────────────────────
        G_osm = to_simple_digraph(G_osm)
        log.info(f"  Simple DiGraph: {G_osm.number_of_nodes()} nodes, "
                 f"{G_osm.number_of_edges()} edges")

        # ── 4. Keep largest weakly-connected component ────────────────────────
        G_osm = filter_to_largest_wcc(G_osm)

        if G_osm.number_of_edges() == 0:
            log.warning(f"  Empty graph after WCC filter – skipping {area}")
            continue

        # ── 5. Compute all-pairs shortest paths (needed for depots) ───────────
        log.info("  Computing all-pairs shortest paths ...")
        dist = dict(nx.all_pairs_dijkstra_path_length(G_osm, weight='weight'))

        # ── 6. Coverage-aware depot placement ─────────────────────────────────
        capacity = float(FLIGHT_BUDGET_S)
        depots   = coverage_aware_depots(G_osm, dist, capacity)
        if not depots:
            # Fallback: highest-degree nodes
            deg    = dict(G_osm.degree())
            n_dep  = max(2, G_osm.number_of_nodes() // 15)
            depots = sorted(deg, key=deg.get, reverse=True)[:n_dep]
            log.warning(f"  Set-cover returned no depots; fallback: {len(depots)} degree-based depots")

        # Enforce minimum 2 depots: add the node farthest from the first depot
        if len(depots) < 2:
            first_depot = depots[0]
            first_dist  = dist.get(first_depot, {})
            non_depots  = [n for n in G_osm.nodes() if n not in depots]
            if non_depots:
                farthest = max(non_depots,
                               key=lambda n: first_dist.get(n, 0.0))
                depots.append(farthest)
                log.info(f"  Added 2nd depot (farthest from depot 1): node {farthest}")

        # ── 7. Compute T_interval ─────────────────────────────────────────────
        total_edge_time = sum(d.get('weight', 0.0) for _, _, d in G_osm.edges(data=True))
        T_interval      = 2.0 * total_edge_time
        recharge_time   = RECHARGE_TIME_S
        log.info(f"  Total edge time: {total_edge_time:.1f}s | "
                 f"T_interval: {T_interval:.1f}s | capacity: {capacity:.0f}s | "
                 f"recharge: {recharge_time:.0f}s | depots: {len(depots)}")

        # ── 8. Load & engineer weather features from raw CSV ──────────────────
        log.info(f"  Loading raw weather from {cfg['csv']} ...")
        try:
            df_area_scaled, avail_bf, valid_node_ids = load_area_features(
                area, cfg['csv'], config, node2idx, scaler_base, base_features
            )
        except FileNotFoundError as e:
            log.error(str(e))
            continue

        if not valid_node_ids:
            log.warning(f"  No matching nodes in global graph for {area} – skipping.")
            continue

        # ── 9. Build per-day full-graph snapshots ─────────────────────────────
        log.info(f"  Building temporal snapshots ...")
        valid_dates, snapshots = build_full_graph_snapshots(
            df_area_scaled, avail_bf, valid_node_ids,
            node2idx, N_global, seq_len, lag_days,
        )
        if not snapshots:
            log.warning(f"  No snapshots for {area} – skipping.")
            continue
        log.info(f"  {len(snapshots)} valid periods (days: {valid_dates[0].date()} "
                 f"– {valid_dates[-1].date()})")

        # ── 10. GNN inference ─────────────────────────────────────────────────
        log.info(f"  Running GNN inference ...")
        all_day_probs = infer_node_probs(model, snapshots, edge_index, edge_weight, device)

        # ── 11. Map node probs → OSM edge probs ──────────────────────────────
        log.info(f"  Mapping probabilities to OSM edges ...")
        edge_probs_all = map_probs_to_edges(G_osm, gnn_node_coords, all_day_probs)
        log.info(f"  Edge probs: {len(edge_probs_all)} unique edges × {len(all_day_probs)} periods")

        # ── 12. Save bundle ───────────────────────────────────────────────────
        with open(graph_pkl, 'wb') as fh:
            pickle.dump(G_osm, fh)

        np.save(edge_npy, edge_probs_all)

        meta = {
            'area':         area,
            'depot_nodes':  depots,
            'capacity':     capacity,
            'recharge_time': recharge_time,
            'T_interval':   T_interval,
            'n_periods':    len(all_day_probs),
            'n_nodes':      G_osm.number_of_nodes(),
            'n_edges':      G_osm.number_of_edges(),
            'tolerance':    cfg['tolerance'],
        }
        np.save(meta_npy, meta)

        log.info(f"  ✓ Saved: {graph_pkl}")
        log.info(f"    {G_osm.number_of_nodes()} nodes | {G_osm.number_of_edges()} edges | "
                 f"{len(depots)} depots | {len(all_day_probs)} periods")

    log.info("\nAll areas processed. Instances in: " + out_root)


if __name__ == '__main__':
    main()
