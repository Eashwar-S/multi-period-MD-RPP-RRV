import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

def visualize_graph(edge_index, edge_labels, title, pos=None):
    """
    Visualizes a graph with edges colored based on labels.
    
    Parameters:
    - edge_index: Tensor of shape [2, num_edges] representing graph edges.
    - edge_labels: Tensor of shape [num_edges] with binary labels (0 or 1).
    - title: Title of the plot (e.g., "Ground Truth Icy Roads").
    - pos: Optional; precomputed layout positions for consistency.
    
    Returns:
    - pos: The layout positions used for the graph.
    """
    # Create a NetworkX graph from edge_index
    G = nx.Graph()
    edges = edge_index.t().tolist()  # Convert edge_index to list of edge tuples
    G.add_edges_from(edges)
    
    # Assign colors to edges based on labels (red for icy, black for normal)
    colors = ['red' if label == 1 else 'black' for label in edge_labels]
    
    # Draw the graph with specified colors
    if pos is None:
        pos = nx.spring_layout(G)  # Compute layout if not provided
    nx.draw(G, pos, node_color='black', edge_color=colors, with_labels=False, 
            node_size=5, font_color='white', width=0.5)
    plt.title(title)
    plt.show()
    return pos

class GNNEdgePredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNEdgePredictor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.3)  # 30% dropout
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout after first convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout after second convolution
        # Edge prediction
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_pred = self.mlp(edge_features).squeeze()
        return torch.sigmoid(edge_pred)  # Probability of icy road
    
class GATEdgePredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GATEdgePredictor, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=True)
        self.gat3 = GATConv(hidden_dim * 4, hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.3)  # 30% dropout
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout after first convolution
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout after second convolution
        x = self.gat3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout after second convolution
		# Edge prediction
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_pred = self.mlp(edge_features).squeeze()
        return torch.sigmoid(edge_pred)  # Probability of icy road
    

class SAGEEdgePredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SAGEEdgePredictor, self).__init__()
        self.sage1 = SAGEConv(input_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.3)  # 30% dropout
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout after first convolution
        x = self.sage2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout after second convolution
        # Edge prediction
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_pred = self.mlp(edge_features).squeeze()
        return torch.sigmoid(edge_pred)  # Probability of icy road

class CombinedGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CombinedGNN, self).__init__()
        # GCN layer
        self.gcn = GCNConv(input_dim, hidden_dim)
        # GAT layer with 4 attention heads
        self.gat = GATConv(hidden_dim, hidden_dim, heads=4, concat=True)
        # GraphSAGE layer (adjust input for GAT's concatenated output)
        self.sage = SAGEConv(hidden_dim * 4, hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.3)  # 30% dropout
        # MLP for edge prediction
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Dropout after GCN
        x = self.gat(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Dropout after GAT
        x = self.sage(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Dropout after GraphSAGE
        # Edge prediction: concatenate embeddings of edge endpoints
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_pred = self.mlp(edge_features).squeeze()
        return torch.sigmoid(edge_pred)



# Training loop
def train(model, loader, optimizer, criterion, epochs=1000):
	scheduler = StepLR(optimizer, step_size=100, gamma=0.5)  # Halve LR every 100 epochs
	model.train()
	losses = []
	for epoch in range(epochs):
		total_loss = 0
		# i = 0
		for data in loader:
			# if i <= 3:
				optimizer.zero_grad()
				out = model(data)  # Edge predictions
				loss = criterion(out, data.edge_label)
				loss.backward()
				optimizer.step()
				total_loss += loss.item()
			# i += 1
		avg_loss = total_loss / 4#len(loader)
		losses.append(avg_loss)

		scheduler.step()  # Update learning rate
		if (epoch + 1) % 50 == 0:
			print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
	return losses

def evaluate_model(model, loader):
	all_preds = []
	all_labels = []
	model.eval()
	i = 0
	with torch.no_grad():
		for data in loader:
			if i >=4:
			# Ground truth labels
				ground_truth_labels = data.edge_label  # Tensor of 0s and 1s
				pred = model(data)
				# print(pred)
				predicted_labels = (pred > 0.7).float()  # Threshold at 0.5
				# print("Predicted icy roads (edge labels):", predicted_labels)
				# Visualize ground truth graph
				pos = visualize_graph(data.edge_index, ground_truth_labels, "Ground Truth Icy Roads")
				
				# Visualize predicted graph with the same layout
				visualize_graph(data.edge_index, predicted_labels, "Predicted Icy Roads", pos=pos)
				all_preds.extend(predicted_labels.numpy())
				all_labels.extend(data.edge_label.numpy())
			i += 1

	test_f1 = f1_score(all_labels, all_preds)
	test_accuracy = accuracy_score(all_labels, all_preds)
	test_precision = precision_score(all_labels, all_preds)
	test_recall = recall_score(all_labels, all_preds)
	print(f'F1-score on test set (192 points, batch size 16): {test_f1:.4f}')
	print(f'Accuracy on test set (192 points, batch size 16): {test_accuracy:.4f}')
	print(f'Precision on test set (192 points, batch size 16): {test_precision:.4f}')
	print(f'Recall on test set (192 points, batch size 16): {test_recall:.4f}')


# Inference example with multiple thresholds and CSV output
def evaluate_and_save_results(model, loader, thresholds=[0.5, 0.6, 0.7, 0.8], output_file='evaluation_results.csv'):
    """
    Evaluate model predictions at different thresholds, compute metrics, and save to CSV.
    
    Parameters:
    - model: Trained GNN model
    - loader: DataLoader with test data
    - thresholds: List of probability thresholds to test
    - output_file: Name of the CSV file to save results
    """
    results = []
    
    model.eval()
    with torch.no_grad():
        all_preds = []  # Store raw probabilities
        all_labels = []  # Store ground truth labels
        
        # Collect all predictions and true labels
        for data in loader:
            # Ground truth labels
            ground_truth_labels = data.edge_label  # Tensor of 0s and 1s
            pred = model(data)  # Raw probabilities
            all_preds.extend(pred.numpy().flatten())
            all_labels.extend(ground_truth_labels.numpy().flatten())
        
        # Convert to numpy arrays for sklearn metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Evaluate for each threshold
        for threshold in thresholds:
            predicted_labels = (all_preds > threshold).astype(float)
            
            # Compute metrics
            accuracy = accuracy_score(all_labels, predicted_labels)
            f1 = f1_score(all_labels, predicted_labels, zero_division=0)
            precision = precision_score(all_labels, predicted_labels, zero_division=0)
            recall = recall_score(all_labels, predicted_labels, zero_division=0)
            
            results.append({
                'Threshold': threshold,
                'Accuracy': accuracy,
                'F1-Score': f1,
                'Precision': precision,
                'Recall': recall
            })
            
            # Optional: Visualize for each threshold (commented out for brevity)
            # predicted_binary = torch.tensor(predicted_labels, dtype=torch.float)
            # pos = visualize_graph(data.edge_index, ground_truth_labels, f"Ground Truth Icy Roads (Threshold {threshold})")
            # visualize_graph(data.edge_index, predicted_binary, f"Predicted Icy Roads (Threshold {threshold})", pos=pos)
        
        # Convert results to DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        print(results_df)
        return results