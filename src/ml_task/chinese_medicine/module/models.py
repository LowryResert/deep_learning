import paddle
import paddle.nn as nn
from paddle.vision.models import MobileNetV1
from paddle.vision.models import MobileNetV2
from paddle.vision.models import ResNet


class MyMobileNetV1(MobileNetV1):
    def forward(self, x, labels=None):
        out = super(MyMobileNetV1, self).forward(x)
        return out if labels is None else calculateAcc(out, labels)


class MyMobileNetV2(MobileNetV2):
    def forward(self, x, labels=None):
        out = super(MyMobileNetV2, self).forward(x)
        return out if labels is None else calculateAcc(out, labels)


class BottleneckBlock(nn.Layer):

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2D(inplanes, width, 1, bias_attr=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv2D(
            width,
            width,
            3,
            padding=dilation,
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias_attr=False)
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2D(
            width, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MyResNet50(ResNet):
    def __init__(self, **kwargs):
        super(MyResNet50, self).__init__(BottleneckBlock, 50, **kwargs)

    def forward(self, x, labels=None):
        out = super(MyResNet50, self).forward(x)
        return out if labels is None else calculateAcc(out, labels)


def calculateAcc(pred, label):
    return pred, paddle.metric.accuracy(input=pred, label=label)
