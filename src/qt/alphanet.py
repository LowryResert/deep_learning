import torch
import torch.nn as nn
from module.component import Cov, Corr, StdDev, ZScore, Return, DecayLinear, Pooling


class AlphaNet(nn.Module):
    """
    width: width of inputted data image
    """
    def __init__(self):
        super(AlphaNet, self).__init__()
        self.ts_cov10 = Cov(9, 30, 10)
        self.ts_corr10 = Corr(9, 30, 10)
        # self.ts_cov10 = Cov(16, 30, 10)
        # self.ts_corr10 = Corr(16, 30, 10)
        self.ts_stddev10 = StdDev(30, 10)
        self.ts_zscore10 = ZScore(30, 10)
        self.ts_return10 = Return(30, 10)
        self.ts_decaylinear10 = DecayLinear(30, 10)
        self.bn = nn.BatchNorm2d(1)
        self.pooling = Pooling(3, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(648, 30)
        # self.fc1 = nn.Linear(1824, 30)
        self.fc2 = nn.Linear(30, 1)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout()

    # x = [batch_size, channels, height, width]
    # [64 * 1 * 9 * 30]
    def forward(self, x):
        # Feature Extraction Layer
        # [64 * 1 * 36 * 3], [64 * 1 * 120 * 3]
        cov10 = self.bn(self.ts_cov10(x))
        corr10 = self.bn(self.ts_corr10(x))
        # [64 * 1 * 9 * 3]
        stddev10 = self.bn(self.ts_stddev10(x))
        zscore10 = self.bn(self.ts_zscore10(x))
        return10 = self.bn(self.ts_return10(x))
        decaylinear10 = self.bn(self.ts_decaylinear10(x))

        # Pooling Layer
        # Concatenate all Features that calculated from different Components
        features = [cov10, corr10, stddev10, zscore10, return10, decaylinear10]
        # [64 * 1 * (36*2+4*9) * 3] = [64 * 324]
        x = torch.cat(features, dim=2)  # [64 * 1 * (120*2+4*9) * 3] = [64 * 828]
        # [64 * 1 * (3 * (36*2+4*9)) * 1] = [64 * 324]]
        x_pooling = self.bn(self.pooling(x))  # [64 * 1 * (3 * (120*2+4*9)) * 1] = [64 * 828]]

        # Fully Connect Layer
        x_flat = self.flatten(x)
        x_pooling_flat = self.flatten(x_pooling)
        # [64 * (2 * 324)]
        y = torch.cat((x_flat, x_pooling_flat), dim=1)  # [64 * (2 * 828)]
        y = self.fc1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.fc2(y)

        return y


# model = AlphaNet()
# t = torch.randint(10, 100, (64, 1, 9, 30)).float()
# pred = model(t)
