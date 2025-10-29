import torch
import torch.nn as nn

class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dropout=0.0):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                                      nn.BatchNorm1d(out_ch))

    def forward(self, x):  # [B, C, T]
        idt = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            idt = self.down(idt)
        out = self.relu(out + idt)
        return out

class ResNet1DClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, channels=(64,128,128,256), kernel_size=5, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, channels[0], kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        layers, in_ch = [], channels[0]
        strides = [1,2,1,2][:len(channels)]
        for i, out_ch in enumerate(channels):
            s = strides[i] if i < len(strides) else 1
            layers += [BasicBlock1D(in_ch, out_ch, kernel_size, stride=s, dropout=dropout),
                       BasicBlock1D(out_ch, out_ch, kernel_size, stride=1, dropout=dropout)]
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x):  # [B, T, D]
        x = x.transpose(1, 2)      # -> [B, D, T]
        x = self.stem(x)
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)
