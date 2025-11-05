# model_def.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, pool=False, drop=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = F.relu(x, inplace=True)
        x = self.pool(x); x = self.drop(x)
        return x

class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 224 -> 112 -> 56 -> 28 -> 14
        self.feat = nn.Sequential(
            ConvBNReLU(3,   32, pool=True,  drop=0.05),
            ConvBNReLU(32,  64, pool=True,  drop=0.05),
            ConvBNReLU(64, 128, pool=True,  drop=0.10),
            ConvBNReLU(128,256, pool=True,  drop=0.10),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.feat(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)