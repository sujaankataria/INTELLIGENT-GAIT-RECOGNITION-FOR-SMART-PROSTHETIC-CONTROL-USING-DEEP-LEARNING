import torch
from torch import nn

class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, conv_channels=64, kernel_size=5, hidden_size=128, lstm_layers=1, dropout=0.3, bidirectional=False):
        super().__init__()
        # Conv over time dimension expects (B, C, T). We will permute in forward.
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(conv_channels, hidden_size, num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers>1 else 0.0, bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.conv(x)          # (B, F, T')
        x = x.permute(0, 2, 1)    # (B, T', F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)
