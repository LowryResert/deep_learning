import paddle
import paddle.nn.functional as F


# 构建模型
class Inception(paddle.nn.Layer):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 路线1，卷积核1x1
        self.route1x1_1 = paddle.nn.Conv2D(in_channels, c1, kernel_size=1)
        # 路线2，卷积层1x1、卷积层3x3
        self.route1x1_2 = paddle.nn.Conv2D(in_channels, c2[0], kernel_size=1)
        self.route3x3_2 = paddle.nn.Conv2D(c2[0], c2[1], kernel_size=3, padding=1)
        # 路线3，卷积层1x1、卷积层5x5
        self.route1x1_3 = paddle.nn.Conv2D(in_channels, c3[0], kernel_size=1)
        self.route5x5_3 = paddle.nn.Conv2D(c3[0], c3[1], kernel_size=5, padding=2)
        # 路线4，池化层3x3、卷积层1x1
        self.route3x3_4 = paddle.nn.MaxPool2D(kernel_size=3, stride=1, padding=1)
        self.route1x1_4 = paddle.nn.Conv2D(in_channels, c4, kernel_size=1)

    def forward(self, x):
        route1 = F.relu(self.route1x1_1(x))
        route2 = F.relu(self.route3x3_2(F.relu(self.route1x1_2(x))))
        route3 = F.relu(self.route5x5_3(F.relu(self.route1x1_3(x))))
        route4 = F.relu(self.route1x1_4(self.route3x3_4(x)))
        out = [route1, route2, route3, route4]
        return paddle.concat(out, axis=1)  # 在通道维度(axis=1)上进行连接


def BasicConv2d(in_channels, out_channels, kernel, stride=1, padding=0):
    layer = paddle.nn.Sequential(
        paddle.nn.Conv2D(in_channels, out_channels, kernel, stride, padding),
        paddle.nn.BatchNorm2D(out_channels, epsilon=1e-3),
        paddle.nn.ReLU())
    return layer


class GoogLeNet(paddle.nn.Layer):
    def __init__(self, in_channel, num_classes):
        super(GoogLeNet, self).__init__()
        self.b1 = paddle.nn.Sequential(
            BasicConv2d(in_channel, out_channels=64, kernel=7, stride=2, padding=3),
            paddle.nn.MaxPool2D(3, 2))
        self.b2 = paddle.nn.Sequential(
            BasicConv2d(64, 64, kernel=1),
            BasicConv2d(64, 192, kernel=3, padding=1),
            paddle.nn.MaxPool2D(3, 2))
        self.b3 = paddle.nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            paddle.nn.MaxPool2D(3, 2))
        self.b4 = paddle.nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            paddle.nn.MaxPool2D(3, 2))
        self.b5 = paddle.nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (182, 384), (48, 128), 128),
            paddle.nn.AvgPool2D(2))
        self.flatten = paddle.nn.Flatten()
        self.b6 = paddle.nn.Linear(9216, num_classes)

    def forward(self, x, label=None):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.flatten(x)
        x = self.b6(x)
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


# module = GoogLeNet(3, 5)
# FLOPs = paddle.flops(module, [1, 3, 224, 224], print_detail=True)
# print(FLOPs)
