import ast
import numpy as np
import torch
from torch_geometric.data import Data

def build_static_graph_structure(df, entity_col='node_id', edge_list_col='edge_list'):
    """
    Given a dataframe representing a static graph (nodes with an edge_list over time),
    extract the unique nodes, build an index mapping, and construct the static edge_index.
    """
    # 1. Map raw entity_ids to contiguous integers [0, N-1]
    unique_nodes = sorted(df[entity_col].unique())
    node2idx = {node: idx for idx, node in enumerate(unique_nodes)}
    
    # 2. Extract edges from one slice of time (since graph is static)
    df_snapshot = df.drop_duplicates(subset=[entity_col]).copy()
    
    coords = {}
    edges = set()
    for _, row in df_snapshot.iterrows():
        u_raw = row[entity_col]
        u = node2idx[u_raw]
        coords[u] = (row.get('latitude', 0.0), row.get('longitude', 0.0))
        
        # Parse edge_list str "[1, 2, 3]" -> list [1, 2, 3]
        try:
            val = row[edge_list_col]
            if isinstance(val, str):
                neighbors_raw = ast.literal_eval(val)
            else:
                neighbors_raw = val
        except:
            neighbors_raw = []
            
        for v_raw in neighbors_raw:
            if v_raw in node2idx:
                v = node2idx[v_raw]
                # Store as directed edges in both directions to make undirected
                edges.add((u, v))
                edges.add((v, u))
                
    # 3. Create edge_index tensor [2, E] and edge_weight tensor [E]
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
    else:
        edges_list = list(edges)
        edge_index_np = np.array(edges_list).T
        edge_index = torch.tensor(edge_index_np, dtype=torch.long).contiguous()
        
        # Calculate inverse distance weights
        weights = []
        for u, v in edges_list:
            lat1, lon1 = coords[u]
            lat2, lon2 = coords[v]
            # Simple Euclidean distance approximation for weight
            dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
            # Inverse distance weighting with small epsilon
            w = 1.0 / (dist + 1e-4)
            weights.append(w)
            
        weights_np = np.array(weights, dtype=np.float32)
        # Normalize weights
        weights_np = weights_np / (np.max(weights_np) + 1e-8)
        edge_weight = torch.tensor(weights_np, dtype=torch.float32)
        
    return node2idx, edge_index, edge_weight

def build_daily_snapshots(df, node2idx, edge_index, date_col='date', feature_cols=None, target_col='icy_label'):
    """
    Given a dataframe grouped by date, creates a list of PyTorch Geometric Data objects.
    Each Data object corresponds to one day, containing node features and edge labels.
    """
    dates = sorted(df[date_col].unique())
    snapshots = []
    
    num_nodes = len(node2idx)
    num_features = len(feature_cols)
    
    # Convert dates to a loop
    for d in dates:
        df_day = df[df[date_col] == d].copy()
        
        # Initialize feature matrix
        x = np.zeros((num_nodes, num_features), dtype=np.float32)
        node_labels = np.zeros(num_nodes, dtype=np.float32)
        
        for _, row in df_day.iterrows():
            idx = node2idx[row['node_id']]
            x[idx, :] = row[feature_cols].values.astype(np.float32)
            node_labels[idx] = float(row[target_col])
            
        # Create edge labels (1 if either endpoint node is icy, else 0)
        # This aligns with the transition from node-level CSVs to edge-level GNN prediction
        node_labels_t = torch.tensor(node_labels, dtype=torch.float32)
        u, v = edge_index[0], edge_index[1]
        
        # OR logic using max
        edge_label = torch.max(node_labels_t[u], node_labels_t[v])
        
        data = Data(
            x=torch.tensor(x, dtype=torch.float32), 
            edge_index=edge_index, 
            edge_label=edge_label,
            date=d
        )
        snapshots.append(data)
        
    return snapshots, dates
