import pandas as pd
import numpy as np
from torch.utils.data import Dataset


def extractData(path):
    df = pd.read_csv(path, dtype={"代码": "category"})
    df = df[df['日期'] >= 20110131]
    df = df[df['日期'] <= 20200529]
    data_list = {'code': df['代码'], 'date': df['日期'], 'return': df['10日回报率'], 'open': df['开盘价(元)'], 'close': df['收盘价(元)'],
            'high': df['最高价(元)'], 'low': df['最低价(元)'], 'volume': df['成交量(股)'], 'vwap': df['vwap/high'] * df['最高价(元)'],
            'turn': df['换手率(%)'], 'free_turn': df['换手率(基准.自由流通股本)'], 'label': df['涨跌幅(%)']}
    data = pd.DataFrame(data_list)

    # Normalization By Feature
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    norm_data = (data - mean) / std

    codes = norm_data.groupby('code').groups.keys()
    data_arr = []
    for code in codes:
        data_arr.append(norm_data[norm_data['code'] == code])
    return data_arr


class Securities(Dataset):
    """
    data: (23400, 16)
    """
    def __init__(self, path='../data/test_data.zip', mode='train'):
        super(Securities, self).__init__()

        org_data = extractData(path)

        self.width = 30  # 数据图片的宽度为30
        training_labels = []
        training_data = []
        valid_labels = []
        valid_data = []
        for sub_data in org_data:
            sub_data.sort_values(by='date', inplace=True)
            feats_data = sub_data.iloc[:, 2: 11].to_numpy(dtype='float32')  # (23400, 9)
            nums, _ = feats_data.shape
            # Labels 预测目标为10天后的涨跌幅
            sub_labels = sub_data.iloc[:, 11].to_numpy(dtype='float32')
            for i in range(self.width + 9, nums):
                start = i - self.width - 9
                end = i - 9
                if i % 2 is not 0:
                    training_labels.append(sub_labels[i])
                    training_data.append(feats_data[start: end])
                else:
                    valid_labels.append(sub_labels[i])
                    valid_data.append(feats_data[start: end])

        if mode == 'train':
            self.labels = training_labels
            self.data = training_data
        elif mode == 'validation':
            self.labels = valid_labels
            self.data = valid_data

    def __getitem__(self, index):
        data_img = self.data[index]
        data_img = data_img.reshape(-1, 30, 9).transpose((0, 2, 1))  # (1, 9, 30)
        data_label = self.labels[index]
        return data_img, np.array([data_label])

    def __len__(self):
        return len(self.labels)
