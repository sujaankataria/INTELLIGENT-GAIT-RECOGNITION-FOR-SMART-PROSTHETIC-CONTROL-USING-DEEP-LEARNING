import torch
from torch import nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size-1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.ReLU(),
            Chomp1d(pad),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.ReLU(),
            Chomp1d(pad),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = self.downsample(x)
        return self.relu(out + res)

class TCNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, channels=(64,64,64,64,64,64), kernel_size=3, dilations=(1,2,4,8,16,32), dropout=0.2):
        super().__init__()
        layers = []
        in_ch = input_dim
        for out_ch, d in zip(channels, dilations):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, d, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.permute(0,2,1)
        y = self.tcn(x)              # (B, F, T)
        y = y[:, :, -1]              # last time step
        return self.head(y)
