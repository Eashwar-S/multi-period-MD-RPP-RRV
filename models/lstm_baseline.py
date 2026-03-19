import torch
import torch.nn as nn

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.3):
        """
        Pure temporal baseline processing sequentially flattened lag features for a single entity.
        Shape expectation at forward: (batch_size, seq_len, input_dim)
        """
        super(LSTMBaseline, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, 1) # Binary classification output

    def forward(self, x):
        # x shape: (N, seq_len, input_dim)
        out, (hn, cn) = self.lstm(x)
        # out shape: (N, seq_len, hidden_dim)
        
        # Take the output of the last time step
        last_out = out[:, -1, :]
        
        y = self.fc1(last_out)
        y = self.relu(y)
        y = self.dropout(y)
        
        logits = self.fc2(y).squeeze(-1)
        return logits
