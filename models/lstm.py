import torch
from torch import nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0, bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)
