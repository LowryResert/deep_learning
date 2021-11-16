import paddle
import paddle.nn.functional as F  # 组网相关的函数，如conv2d, relu...
from paddle import nn

# 构建模型
class Residual(paddle.nn.Layer):
    def __init__(self, in_channel, out_channel, use_conv1x1=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2D(out_channel, out_channel, kernel_size=3, padding=1)
        if use_conv1x1:  # 使用1x1卷积核
            self.conv3 = nn.Conv2D(in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.batchNorm1 = nn.BatchNorm2D(out_channel)
        self.batchNorm2 = nn.BatchNorm2D(out_channel)

    def forward(self, x):
        y = F.relu(self.batchNorm1(self.conv1(x)))
        y = self.batchNorm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        out = F.relu(y + x)  # 核心代码
        return out


def ResNetBlock(in_channel, out_channel, num_layers, is_first=False):
    if is_first:
        assert in_channel == out_channel
    block_list = []
    for i in range(num_layers):
        if i == 0 and not is_first:
            block_list.append(Residual(in_channel, out_channel, use_conv1x1=True, stride=2))
        else:
            block_list.append(Residual(out_channel, out_channel))
    resNetBlock = nn.Sequential(*block_list)  # 用*号可以把list列表展开为元素
    return resNetBlock


class ResNet50(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1))
        self.b2 = ResNetBlock(64, 64, 3, is_first=True)
        self.b3 = ResNetBlock(64, 128, 4)
        self.b4 = ResNetBlock(128, 256, 6)
        self.b5 = ResNetBlock(256, 512, 3)
        self.AvgPool = nn.AvgPool2D(2)
        self.flatten = nn.Flatten()
        self.Linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.AvgPool(x)
        x = self.flatten(x)
        x = self.Linear(x)
        return x


resnet = ResNet50(num_classes=5)
paddle.summary(resnet, (-1, 3, 224, 224))
