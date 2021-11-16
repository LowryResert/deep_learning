import paddle
import paddle.nn as nn
import numpy as np


# GRU网络
class MyModelV1(nn.Layer):
    def __init__(self, vocab_size):
        super(MyModelV1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.gru = nn.GRU(256, 256, num_layers=2, direction='bidirectional', dropout=0.5)
        self.linear = nn.Linear(in_features=256 * 2, out_features=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        emb = self.dropout(self.embedding(inputs))
        # output形状大小为[batch_size,seq_len,num_directions * hidden_size]
        # hidden形状大小为[num_layers * num_directions, batch_size, hidden_size]
        # 把前向的hidden与后向的hidden合并在一起
        output, hidden = self.gru(emb)
        hidden = paddle.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)
        # hidden形状大小为[batch_size, hidden_size * num_directions]
        hidden = self.dropout(hidden)
        return self.linear(hidden)


# RNN
# 定义RNN网络
class MyModelV2(nn.Layer):
    def __init__(self, vocab_size):
        super(MyModelV2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.rnn = nn.SimpleRNN(256, 256, num_layers=2, direction='forward', dropout=0.5)
        self.linear = nn.Linear(in_features=256 * 2, out_features=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        emb = self.dropout(self.embedding(inputs))
        # output形状大小为[batch_size,seq_len,num_directions * hidden_size]
        # hidden形状大小为[num_layers * num_directions, batch_size, hidden_size]
        # 把前向的hidden与后向的hidden合并在一起
        output, hidden = self.rnn(emb)
        hidden = paddle.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)
        # hidden形状大小为[batch_size, hidden_size * num_directions]
        hidden = self.dropout(hidden)
        return self.linear(hidden)
