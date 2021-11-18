import pandas as pd
import numpy as np
from torch.utils.data import Dataset


keys = ['code', 'date', 'return', 'open', 'close', 'high', 'low', 'volume', 'vwap', 'turn', 'free_turn', 'label']


def extractData(path):
    df = pd.read_csv(path, dtype={"代码": "category"})
    df = df[df['日期'] >= 20110131]
    df = df[df['日期'] <= 20200529]
    list = [df['代码'], df['日期'], df['10日回报率'], df['开盘价(元)'], df['收盘价(元)'],
            df['最高价(元)'], df['最低价(元)'], df['成交量(股)'], df['vwap/high'] * df['最高价(元)'],
            df['换手率(%)'], df['换手率(基准.自由流通股本)'], df['涨跌幅(%)']]
    # data_list = {'code': df['代码'], 'date': df['日期'], 'return': df['10日回报率'], 'open': df['开盘价(元)'],
    #              'close': df['收盘价(元)'], 'high': df['最高价(元)'], 'low': df['最低价(元)'], 'volume': df['成交量(股)'],
    #              'vwap': df['vwap/high'] * df['最高价(元)'], 'turn': df['换手率(%)'],
    #              'free_turn': df['换手率(基准.自由流通股本)'], 'label': df['涨跌幅(%)']}
    data = np.array(list)

    data_to_norm = data[2:11]
    data_to_norm = data_to_norm.astype('float32')
    # Normalization By Feature
    mean = data_to_norm.mean(axis=1, keepdims=True)
    std = data_to_norm.std(axis=1, keepdims=True)
    normed_data = (data_to_norm - mean) / std
    data = np.vstack((data[0:2], normed_data, data[11]))

    # Compress as DataFrame
    data = data.T
    data = pd.DataFrame(data, columns=keys)

    codes = data.groupby('code').groups.keys()
    data_arr = []
    for code in codes:
        data_arr.append(data[data['code'] == code])
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
                # for i in range(self.width + 9, 100):
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
