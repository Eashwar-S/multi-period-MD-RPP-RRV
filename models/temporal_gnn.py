import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class TemporalNodeGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, dropout=0.3, num_layers=2, gnn_type='GCN', temporal_model='gru'):
        """
        Temporal GNN for dynamic features on static edges (Node Classification).
        1. Temporal aggregator (GRU/LSTM) over time for each node.
        2. GNN spatial convolution on the final hidden state using edge weights.
        3. Node-level binary classification head with skip connections.
        """
        super(TemporalNodeGNN, self).__init__()
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
        
        # 3. Node Prediction Head (Skip connection: raw temporal + message passing)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, edge_index, edge_weight=None):
        if x.dim() == 2:
            try:
                x = x.view(x.shape[0], self.seq_len, self.input_dim)
            except Exception as e:
                raise RuntimeError(f"Could not reshape input {x.shape} for seq_len {self.seq_len} and input_dim {self.input_dim}") from e
                
        # 1. Temporal aggregation per node
        out, _ = self.temporal(x)
        h_temp = out[:, -1, :] # Shape: (num_nodes, hidden_dim)
        
        # 2. Spatial Convolution
        if self.gnn_type == 'SAGE':
             # PyG SAGEConv does not cleanly accept edge_weight out of the box without Custom classes, so we safely omit for SAGE or handle it based on version.
             h = self.conv1(h_temp, edge_index)
        else:
             h = self.conv1(h_temp, edge_index, edge_weight)
        
        h = F.relu(h)
        h = self.dropout(h)
        
        if self.gnn_type == 'SAGE':
             h = self.conv2(h, edge_index)
        else:
             h = self.conv2(h, edge_index, edge_weight)
             
        h = F.relu(h)
        h = self.dropout(h)
        
        # 3. Node level prediction (Skip Connection)
        node_features = torch.cat([h_temp, h], dim=1) # (num_nodes, 2 * hidden_dim)
        logits = self.mlp(node_features).squeeze(-1)
        return logits

class StaticNodeGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3, gnn_type='GCN'):
        super(StaticNodeGNN, self).__init__()
        self.dropout_rate = dropout
        self.gnn_type = gnn_type
        
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
        
    def forward(self, x, edge_index, edge_weight=None):
        if x.dim() == 3:
            x = x.view(x.shape[0], -1)
            
        h_in = F.relu(self.fc_in(x))
        h_in = self.dropout(h_in)
        
        if self.gnn_type == 'SAGE':
             h = self.conv1(h_in, edge_index)
        else:
             h = self.conv1(h_in, edge_index, edge_weight)
             
        h = F.relu(h)
        h = self.dropout(h)
        
        if self.gnn_type == 'SAGE':
             h = self.conv2(h, edge_index)
        else:
             h = self.conv2(h, edge_index, edge_weight)
             
        h = F.relu(h)
        h = self.dropout(h)
        
        node_features = torch.cat([h_in, h], dim=1)
        logits = self.mlp(node_features).squeeze(-1)
        return logits
