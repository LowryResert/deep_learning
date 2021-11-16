import torch
import torch.nn as nn
# import pandas as pd
# from dataset import Securities
# from torch.utils.data import DataLoader


def generate_steps(width: int, stride: int) -> list:
    if stride <= 0:
        raise ValueError("stride should be positive but got {}.".format(stride))
    steps = []
    start = 0
    while start < width:
        steps.append(start)
        start += stride
    steps.append(width)
    return steps


def generate_steps_pair(feat_num):
    x = []
    y = []
    for i in range(feat_num):
        for j in range(i + 1, feat_num):
            x.append(i)
            y.append(j)
    return x, y


class Cov(nn.Module):

    def __init__(self, height, width, stride):
        super(Cov, self).__init__()
        self.pair_x, self.pair_y = generate_steps_pair(height)
        self.steps = generate_steps(width, stride)

    def forward(self, input):
        if input.dim() != 4:
            raise ValueError("expected 2D or 3D input (got {}D input)".format(input.dim()))
        steps = self.steps
        covs = []
        for i in range(len(steps) - 1):
            start = steps[i]
            end = steps[i + 1]
            x = input[:, :, self.pair_x, start:end]
            y = input[:, :, self.pair_y, start:end]
            x_mean = x.mean(dim=3, keepdims=True)
            y_mean = y.mean(dim=3, keepdims=True)
            x_sp = x - x_mean
            y_sp = y - y_mean
            cov = ((x_sp * y_sp) / (end - start - 1)).sum(dim=3, keepdims=True)
            covs.append(cov)
        return torch.cat(covs, dim=3)


class Corr(nn.Module):

    def __init__(self, height, width, stride):
        super(Corr, self).__init__()
        self.pair_x, self.pair_y = generate_steps_pair(height)
        self.steps = generate_steps(width, stride)

    def forward(self, input):
        if input.dim() != 4:
            raise ValueError("expected 2D or 3D input (got {}D input)".format(input.dim()))
        steps = self.steps
        stds = []
        for i in range(len(steps) - 1):
            start = steps[i]
            end = steps[i + 1]
            x = input[:, :, self.pair_x, start:end]
            y = input[:, :, self.pair_y, start:end]
            x_mean = x.mean(dim=3, keepdims=True)
            y_mean = y.mean(dim=3, keepdims=True)
            cov = (((x - x_mean) * (y - y_mean)) / (end - start - 1))
            x_std = x.std(dim=3, keepdims=True)
            y_std = y.std(dim=3, keepdims=True)
            std = ((cov / (x_std * y_std)) / (end - start - 1)).sum(dim=3, keepdims=True)
            stds.append(std)
        return torch.cat(stds, dim=3)


class StdDev(nn.Module):
    """
    width: 数据图片的宽度
    stride: 步长
    """

    def __init__(self, width, stride):
        super(StdDev, self).__init__()
        self.steps = generate_steps(width, stride)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("expected 2D or 3D input (got {}D input)".format(x.dim()))
        steps = self.steps
        stds = []
        for i in range(len(steps) - 1):
            start = steps[i]
            end = steps[i + 1]
            sub_x = x[:, :, :, start:end]
            std_sub_x = sub_x.std(dim=3, keepdims=True)
            stds.append(std_sub_x)
        return torch.cat(stds, dim=3)


class ZScore(nn.Module):

    def __init__(self, width, stride):
        super(ZScore, self).__init__()
        self.steps = generate_steps(width, stride)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("expected 2D or 3D input (got {}D input)".format(x.dim()))
        steps = self.steps
        zscores = []
        for i in range(len(steps) - 1):
            start = steps[i]
            end = steps[i + 1]
            sub_x = x[:, :, :, start:end]
            std_sub_x = sub_x.std(dim=3, keepdims=True)
            mean_sub_x = sub_x.mean(dim=3, keepdims=True)
            zscore_sub_x = mean_sub_x / std_sub_x
            zscores.append(zscore_sub_x)
        return torch.cat(zscores, dim=3)


class Return(nn.Module):

    def __init__(self, width, stride):
        super(Return, self).__init__()
        self.steps = generate_steps(width, stride)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("expected 2D or 3D input (got {}D input)".format(x.dim()))
        steps = self.steps
        returns = []
        for i in range(len(steps) - 1):
            start = steps[i]
            end = steps[i + 1]
            sub_x = x[:, :, :, start:end]
            x_today = sub_x.narrow_copy(3, -1, 1)
            x_pre = sub_x.narrow_copy(3, 0, 1)
            return_sub_x = x_today / x_pre - 1
            returns.append(return_sub_x)
        return torch.cat(returns, dim=3)


class DecayLinear(nn.Module):

    def __init__(self, width, stride):
        super(DecayLinear, self).__init__()
        self.steps = generate_steps(width, stride)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("expected 2D or 3D input (got {}D input)".format(x.dim()))
        steps = self.steps
        decays = []
        for i in range(len(steps) - 1):
            start = steps[i]
            end = steps[i + 1]
            sub_x = x[:, :, :, start:end]
            time_range = end - start
            weight = torch.arange(1, time_range + 1)
            weight = weight / weight.sum()
            decay_sub_x = (sub_x * weight).sum(dim=3, keepdims=True)
            decays.append(decay_sub_x)
        return torch.cat(decays, dim=3)


class Pooling(nn.Module):

    def __init__(self, width, stride):
        super(Pooling, self).__init__()
        self.steps = generate_steps(width, stride)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("expected 2D or 3D input (got {}D input)".format(x.dim()))
        steps = self.steps
        pooling_res = []
        for i in range(len(steps) - 1):
            start = steps[i]
            end = steps[i + 1]
            sub_x = x[:, :, :, start:end]
            mean = sub_x.mean(dim=3, keepdims=True)
            max = sub_x.max(dim=3, keepdims=True).values
            min = sub_x.min(dim=3, keepdims=True).values
            y = torch.cat((mean, max, min), dim=2)
            pooling_res.append(y)
        return torch.cat(pooling_res, dim=3)


# securities = Securities()
# loader = DataLoader(data, batch_size=5)
# path = '../data/test_data.zip'
# df = pd.read_csv(path, dtype={"代码": "category"})
# df = df[df['日期'] >= 20110131]
# df = df[df['日期'] <= 20200529]
# codes = list(df.groupby('代码').groups.keys())
# df1 = df[df['代码'] == codes[0]]
# df = pd.read_csv('../img_data_nan.txt', header=None, sep=' ')
# x = df.to_numpy(dtype='float32')
# x = x.reshape((1, 1, 16, 30))
# x = torch.from_numpy(x)
