import paddle
import matplotlib.pyplot as plt
import paddle.vision.transforms as T
import paddle.nn.functional as F

print(paddle.__version__)
transform = T.Compose([T.Normalize(mean=[127.5],
                                   std=[127.5],
                                   data_format='CHW')])

# 对数据进行归一化
train_dataset = paddle.vision.datasets.FashionMNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.FashionMNIST(mode='test', transform=transform)

# 随机选取十个训练集数据可视化
label_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(6, 9))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = paddle.randint(len(train_dataset), shape=(1,)).item()
    img, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label_map[label[0]])
    plt.axis('off')
    # numpy.squeeze 删除长度为1的轴 (1, 28, 28) =>(28, 28)
    plt.imshow(img.squeeze(), cmap='gray')

plt.show()

# 随机选取一个测试数据进行可视化
sample_idx = paddle.randint(len(test_dataset), shape=(1,)).item()
img, label = test_dataset[sample_idx]
plt.imshow(img.squeeze(), cmap='gray')
plt.show()
print(f"Label: {label_map[label.item()]}")


# 自己实现的网络
class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=3, stride=2)
        self.conv3 = paddle.nn.Conv2D(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0)
        self.linear1 = paddle.nn.Linear(in_features=128, out_features=100)
        self.linear2 = paddle.nn.Linear(in_features=100, out_features=64)
        self.linear3 = paddle.nn.Linear(in_features=64, out_features=10)
        self.flatten = paddle.nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


net_cls = MyNet()
paddle.summary(net_cls, (-1, 1, 28, 28))


# 先实现LeNet，随后测试LeNet的性能
class LeNet(paddle.nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=3, stride=2)
        self.flatten = paddle.nn.Flatten()
        self.linear1 = paddle.nn.Linear(in_features=256, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


# net_cls = LeNet()
# paddle.summary(net_cls, (-1, 1, 28, 28))

# 用Model封装模型
model = paddle.Model(net_cls)

# 参数
lr = 0.001
epoch = 1
batch_size = 64
log_dir = '../log/train'  # 训练过程中的loss变化log保存路径
save_dir = 'log_output/module'  # 模型保存路径

# 损失函数
loss_fn = paddle.nn.CrossEntropyLoss()
# 优化函数
optimizer = paddle.optimizer.SGD(learning_rate=lr, parameters=model.parameters())

# 配置模型
model.prepare(optimizer, loss_fn, paddle.metric.Accuracy())

# 利用visualdl可视化，然后训练并保存模型
callback = paddle.callbacks.VisualDL(log_dir=log_dir)

model.fit(train_dataset, test_dataset, epochs=epoch, batch_size=batch_size, log_freq=1000, save_dir=save_dir,
          callbacks=callback)


class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
        )
        self.flatten = paddle.nn.Flatten()
        self.fc = paddle.nn.Sequential(
            paddle.nn.Linear(128, 64),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(0.5),
            paddle.nn.Linear(64, 32),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(0.5),
            paddle.nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


net_cls = MyNet()
paddle.summary(net_cls, (-1, 1, 28, 28))


class MyNet_1(paddle.nn.Layer):
    def __init__(self):
        super(MyNet_1, self).__init__()
        self.conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.BatchNorm2D(64),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            paddle.nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            paddle.nn.MaxPool2D(kernel_size=2, stride=2, padding=1),
            paddle.nn.BatchNorm2D(128),
            paddle.nn.ReLU()
        )
        self.flatten = paddle.nn.Flatten()
        self.fc = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=128*8*8, out_features=512),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(),
            paddle.nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x



