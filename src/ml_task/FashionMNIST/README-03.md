# 机器学习作业03 - Fashion Mnist实现图像分类

姓名：罗睿卿

学号：21214935

## 约定
- 为了方便描述，此文档中约定：HOME=/home/aistudio

- 由于aistudio平台对上传文件大小的限制，我只保留最终采用的模型参数，对于中间结果全部删除。

## 流程

### 预处理

使用paddle.vision.transform API对下载之后的训练集和测试集进行归一化处理，使得每个样本变成单通道28*28的灰度图（1，28，28）。

### 样本的可视化

对训练集和测试集样本进行可视化：
- 随机选取了十个训练集中的数据及对应标签进行可视化，结果放在 $HOME/visualization/data/training_data.png；
- 随机选取一个测试集中的数据进行可视化，结果放在 $HOME/visualization/data/test_data.png。

### 在LeNet上训练以获取性能基准

使用LeNet作为性能基准，训练并可视化。训练集中有60000个样本，每个batch有64个样本，共938个batch，训练共有50个epoch。可视化内容为：
- 训练集的loss曲线变化和acc变化，横坐标为训练样本的index；
- 测试集的loss曲线变化和acc变化，横坐标为epoch的index；

可视化结果保存在 $HOME/visualization/lenet 文件夹下。正如在notebook中展示的那样，我使用LeNet的最终训练结果，在测试集上（10000个样本）的正确率为**87.44%**。最终的模型参数保存在 $HOME/output/lenet_model/final.pdparams中。

### 定义模型

LeNet本身是一个非常经典的卷积神经网络模型，其在面对单通道图像分类任务时性能已经非常优越。正如我前面所做的那样，使用Lenet在FashionMNIST数据集上已经有了87.44%的正确率。

在改进LeNet在此数据集上的性能时，主要有以下几个改进点：

1. 加深模型的深度，增加卷积层或全连接层。
2. 增加参数数量。把图像由单个特征（单通道）变为多个特征。
3. 使用dropout。去掉一些参数，减小过拟合。

### 训练模型

与LeNet训练过程一样，自定义模型也分为50个epoch。60000个训练样本分为938个batch，每个batch有64个样本。最终训练结果保存在 $HOME/output/model/final.pdparams 中。

### 测试模型

经过10000个测试数据的测试，自定义的模型正确率达到了**91.59%**。

### 可视化

自定义模型的可视化结果保存在$HOME/visualization/mynet文件夹下。包括：

1. loss值在训练集和测试集上的变化；
2. 正确率在训练集和测试集上的变化。

预测结果正确率保存为eval_acc.png。横坐标为当前epoch的index，纵坐标为当前epoch时参数的正确率。可以看到，当第50个epoch时正确率最高，达到了**91.59%**。
