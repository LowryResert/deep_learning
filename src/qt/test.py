import torch
import torch.nn as nn
import pandas as pd
from alphanet import AlphaNet


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

# df = df[df['日期'] >= 20110131]
# df = df[df['日期'] <= 20200529]
# df.groupby('代码').count()

df.sort_values(by=['代码', '日期'], inplace=True)
