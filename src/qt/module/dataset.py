import pandas as pd
import numpy as np
from torch.utils.data import Dataset


def extractData(path):
    data = []
    df = pd.read_csv(path, dtype={"代码": "category"})
    df = df[df['日期'] >= 20110131]
    df = df[df['日期'] <= 20200529]
    codes = df.groupby('代码').groups.keys()
    for code in codes:
        data.append(df[df['代码'] == code])
    return data


class Securities(Dataset):
    """
    data: (23400, 16)
    """
    def __init__(self, path, mode='train'):
        super(Securities, self).__init__()

        org_data = extractData(path)

        self.width = 30  # 数据图片的宽度为30
        training_labels = []
        training_data = []
        valid_labels = []
        valid_data = []
        for sub_data in org_data:
            sub_data.sort_values(by='日期', inplace=True)
            sub_data = sub_data.iloc[:, 2: 18].to_numpy(dtype='float64')  # (23400, 16)
            nums, self.height = sub_data.shape
            # Labels 预测目标为5天后的涨跌幅
            sub_labels = sub_data[:, 5]
            for i in range(self.width + 4, nums):
                start = i - self.width - 4
                end = i - 4
                if i % 2 is 0:
                    training_labels.append(sub_labels[i])
                    training_data.append(sub_data[start: end])
                else:
                    valid_labels.append(sub_labels[i])
                    valid_data.append(sub_data[start: end])

        if mode == 'train':
            self.labels = training_labels
            self.data = training_data
        elif mode == 'validation':
            self.labels = valid_labels
            self.data = valid_data

    def __getitem__(self, index):
        data_img = self.data[index]
        data_img = data_img.reshape(-1, 30, 16).transpose((0, 2, 1))  # (1, 16, 30)
        data_label = self.labels[index]
        return data_img, np.array([data_label])

    def __len__(self):
        return len(self.labels)
