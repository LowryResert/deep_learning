import numpy as np
import paddle
from models import MyModelV1, MyModelV2


version = 'v4'
# 准备数据
# 加载IMDB数据
imdb_train = paddle.text.datasets.Imdb(mode='train')  # 训练数据集
imdb_test = paddle.text.datasets.Imdb(mode='test')  # 测试数据集
# 获取字典
word_dict = imdb_train.word_idx
# 在字典中增加一个<pad>字符串
word_dict['<pad>'] = len(word_dict)
# 参数设定
vocab_size = len(word_dict)
embedding_size = 256
hidden_size = 256
n_layers = 2
dropout = 0.5
seq_len = 200
batch_size = 64
epochs = 10
lr = 0.0001
log_dir = 'output/' + version
save_dir = 'work/' + version
pad_id = word_dict['<pad>']


# 每个样本的单词数量不一样，用Padding使得每个样本输入大小为seq_len
def padding(dataset):
    padded_sents = []
    labels = []
    for batch_id, data in enumerate(dataset):
        sent, label = data[0].astype('int64'), data[1].astype('int64')
        padded_sent = np.concatenate([sent[:seq_len], [pad_id] * (seq_len - len(sent))]).astype('int64')
        padded_sents.append(padded_sent)
        labels.append(label)
    return np.array(padded_sents), np.array(labels)


train_x, train_y = padding(imdb_train)
test_x, test_y = padding(imdb_test)


class IMDBDataset(paddle.io.Dataset):
    def __init__(self, sents, labels):
        super().__init__()
        self.sents = sents
        self.labels = labels

    def __getitem__(self, index):
        data = self.sents[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.sents)


train_dataset = IMDBDataset(train_x, train_y)
test_dataset = IMDBDataset(test_x, test_y)

train_loader = paddle.io.DataLoader(train_dataset, return_list=True, shuffle=True, batch_size=batch_size,
                                    drop_last=True)
test_loader = paddle.io.DataLoader(test_dataset, return_list=True, shuffle=True, batch_size=batch_size, drop_last=True)


# 封装模型
myNet = MyModelV1(vocab_size)
# myNet = MyModelV2(vocab_size)
model = paddle.Model(myNet)  # 用Model封装模型

loss_fn = paddle.nn.CrossEntropyLoss()
optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())

# 配置模型优化器、损失函数、评估函数
model.prepare(optimizer, loss_fn, paddle.metric.Accuracy())

callback = paddle.callbacks.VisualDL(log_dir=log_dir)

# 模型训练
model.fit(train_loader, test_loader,
          epochs=epochs, batch_size=batch_size, save_dir=save_dir, callbacks=callback)

# 模型评估
model_state_dict = paddle.load(save_dir + '/final.pdparams')  # 导入最优模型
myNet = MyModelV1(vocab_size)
# myNet = MyModelV2(vocab_size)
myNet.set_state_dict(model_state_dict)
model = paddle.Model(myNet)
model.prepare(loss=loss_fn, metrics=paddle.metric.Accuracy())
eval_result = model.evaluate(test_loader, batch_size=batch_size, verbose=1)
print(eval_result)
