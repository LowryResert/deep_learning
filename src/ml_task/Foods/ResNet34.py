import paddle
from paddle import nn
from paddle.nn import functional as f


# 定义Residual Block。左边包含Conv、BatchNorm、ReLU；右边是一个shortcut。
class ResidualBlock(nn.Layer):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2D(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2D(out_channel),
            nn.ReLU(),
            nn.Conv2D(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(out_channel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return f.relu(out)


# 定义ResNet34网络。初始为普通卷积结构，
# layer1中含有3个residual，layer2中含有4个residual，layer3中含有6个residual，layer4中含有3个residual
# 最后有一个全连接层
class ResNet(nn.Layer):
    """
    实现主ResNet34。
    ResNet34包含多个layer，每个layer又包含多个residual block
    用_make_layer函数来实现4个residual layer
    """
    def __init__(self, num_classes=5):
        super(ResNet, self).__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        )
        # 用_make_layer实现重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)
        # 全连接层
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=4608, out_features=num_classes)

    def _make_layer(self, in_channel, out_channel, block_num, stride=1):
        # 构造layer，包含多个residual block
        shortcut = nn.Sequential(
            nn.Conv2D(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride),
            nn.BatchNorm2D(out_channel)
        )
        layers = [ResidualBlock(in_channel, out_channel, stride, shortcut)]
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        # normal convolution
        x = self.pre(inputs)

        # 4 residual blocks
        x = self.layer1(x)
        x = self.layer1(x)
        x = self.layer1(x)
        x = self.layer1(x)

        # fully connect
        x = f.avg_pool2d(x, 7)
        x = self.flatten(x)
        return self.fc(x)


net = ResNet()
paddle.summary(net, (-1, 3, 224, 224))
