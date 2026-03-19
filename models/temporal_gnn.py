import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class TemporalEdgeGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, dropout=0.3, num_layers=2, gnn_type='GCN', temporal_model='gru'):
        """
        Temporal GNN for dynamic features on static edges.
        1. Temporal aggregator (GRU/LSTM) over time for each node.
        2. GNN spatial convolution on the final hidden state.
        3. Edge-level binary classification head using concatenated embeddings.
        """
        super(TemporalEdgeGNN, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        # 1. Temporal Aggregator
        if temporal_model.lower() == 'gru':
            self.temporal = nn.GRU(input_dim, hidden_dim, batch_first=True)
        else:
            self.temporal = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            
        # 2. Spatial Graph layers
        self.gnn_type = gnn_type
        if gnn_type == 'GCN':
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == 'GAT':
            self.conv1 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        elif gnn_type == 'SAGE':
            self.conv1 = SAGEConv(hidden_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        else:
            raise ValueError("gnn_type must be GCN, GAT, or SAGE")
            
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # 3. Edge Prediction Head
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, edge_index):
        """
        x: shape (num_nodes, seq_len * input_dim) or already (num_nodes, seq_len, input_dim)
        edge_index: shape (2, num_edges)
        """
        if x.dim() == 2:
            try:
                # Reshape flattened lagging features to sequence 
                # e.g., if there are 15 base features and 3 lags + 1 current = 4 steps => 60 cols
                x = x.view(x.shape[0], self.seq_len, self.input_dim)
            except Exception as e:
                raise RuntimeError(f"Could not reshape input {x.shape} for seq_len {self.seq_len} and input_dim {self.input_dim}") from e
                
        # 1. Temporal aggregation per node
        # x is (num_nodes, seq_len, input_dim). We process it individually per node.
        out, _ = self.temporal(x)
        # out is (num_nodes, seq_len, hidden_dim). Take the last time step.
        h = out[:, -1, :] # Shape: (num_nodes, hidden_dim)
        
        # 2. Spatial Convolution
        h = self.conv1(h, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        
        # 3. Edge level prediction
        u, v = edge_index[0], edge_index[1]
        edge_features = torch.cat([h[u], h[v]], dim=1) # (num_edges, 2 * hidden_dim)
        
        logits = self.mlp(edge_features).squeeze(-1)
        return logits

class StaticEdgeGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3, gnn_type='GCN'):
        """
        Ablation without the temporal sequence recurrent model.
        Simply treats all input dimensions (flattened lags) as standard node features.
        """
        super(StaticEdgeGNN, self).__init__()
        self.dropout_rate = dropout
        
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        
        if gnn_type == 'GCN':
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == 'GAT':
            self.conv1 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        elif gnn_type == 'SAGE':
            self.conv1 = SAGEConv(hidden_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, edge_index):
        if x.dim() == 3:
            # Flatten if passed as 3D (num_nodes, seq_len, feat_dim)
            x = x.view(x.shape[0], -1)
            
        h = F.relu(self.fc_in(x))
        h = self.dropout(h)
        
        h = self.conv1(h, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        
        u, v = edge_index[0], edge_index[1]
        edge_features = torch.cat([h[u], h[v]], dim=1)
        
        logits = self.mlp(edge_features).squeeze(-1)
        return logits
