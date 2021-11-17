import torch
import torch.nn as nn
from alphanet import AlphaNet
from module.component import Cov, Corr, StdDev, ZScore, Return, DecayLinear, Pooling
import pandas as pd


# df = pd.read_csv('data/data.csv', header=None)
# print(df)
#
# data = df.to_numpy(dtype='float32').reshape(-1, 1, 16, 30)
# t_data = torch.from_numpy(data)
# model = AlphaNet()
#
# pred = model(t_data)


path = './data/test_data.zip'
df = pd.read_csv(path, dtype={"代码": "category"})
# org_data = df.iloc[:, 2: 18].to_numpy(dtype='float64')  # (23400, 16)

df = df[df['日期'] >= 20110131]
df = df[df['日期'] <= 20200529]
# df.groupby('代码').count()
# df.sort_values(by=['代码', '日期'], inplace=True)

# x = pd.read_csv('x.txt', header=None, sep=' ')
# x = x.to_numpy()
# x_t = x.reshape(1, 1, 9, 30)
# t = torch.tensor(x_t)
#
# cov = Cov(2, 30, 10)
# c1 = cov(t[:, :, :2, :])
# c2 = cov(t[:, :, [0, 5], :])
