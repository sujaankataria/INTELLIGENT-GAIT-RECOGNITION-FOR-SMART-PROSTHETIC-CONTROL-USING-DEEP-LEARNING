import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, bidirectional, dropout, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):  # [B, T, D]
        out, h = self.gru(x)  # h: [layers*dirs, B, H]
        if self.gru.bidirectional:
            h_last = torch.cat([h[-2], h[-1]], dim=1)
        else:
            h_last = h[-1]
        return self.fc(self.dropout(h_last))
