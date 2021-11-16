import os
import zipfile
import random
import json
import paddle
import numpy as np
from PIL import Image
from paddle.metric import Accuracy

'''
参数配置
'''
train_parameters = {
    "version": "net1",
    "input_size": [3, 224, 224],  # 输入图片的shape
    "class_dim": -1,  # 分类数
    "src_path": "./data/data42610/foods.zip",  # 原始数据集路径
    "target_path": "./data/",  # 要解压的路径
    "train_list_path": "./data/train.txt",  # train.txt路径
    "eval_list_path": "./data/eval.txt",  # eval.txt路径
    "readme_path": "./data/readme.json",  # readme.json路径
    "log_dir": "./log/",
    "save_dir": "./module/",
    "label_dict": {},  # 标签字典
    "num_epochs": 1,  # 训练轮数
    "batch_size": 64,  # 训练时每个批次的大小
    "learning_strategy": {  # 优化函数相关的配置
        "lr": 0.001  # 超参数学习率
    }
}


def unzip_data(src_path, target_path):
    """
    解压原始数据集，将src_path路径下的zip包解压至target_path目录下
    """
    if not os.path.isdir(target_path + "foods"):
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()


def get_data_list(target_path, train_list_path, eval_list_path):
    """
    生成数据列表
    """
    # 存放所有类别的信息
    class_detail = []
    # 获取所有类别保存的文件夹名称
    data_list_path = target_path + "foods/"
    class_dirs = os.listdir(data_list_path)
    # 总的图像数量
    all_class_images = 0
    # 存放类别标签
    class_label = 0
    # 存放类别数目
    class_dim = 0
    # 存储要写进eval.txt和train.txt中的内容
    trainer_list = []
    eval_list = []
    # 读取每个类别
    for class_dir in class_dirs:
        if class_dir != ".DS_Store":
            class_dim += 1
            # 每个类别的信息
            class_detail_list = {}
            eval_sum = 0
            trainer_sum = 0
            # 统计每个类别有多少张图片
            class_sum = 0
            # 获取类别路径
            path = data_list_path + class_dir
            # 获取所有图片
            img_paths = os.listdir(path)
            for img_path in img_paths:  # 遍历文件夹下的每个图片
                name_path = path + '/' + img_path  # 每张图片的路径
                if class_sum % 8 == 0:  # 每8张图片取一个做验证数据
                    eval_sum += 1  # test_sum为测试数据的数目
                    eval_list.append(name_path + "\t%d" % class_label + "\n")
                else:
                    trainer_sum += 1
                    trainer_list.append(name_path + "\t%d" % class_label + "\n")  # trainer_sum测试数据的数目
                class_sum += 1  # 每类图片的数目
                all_class_images += 1  # 所有类图片的数目

            # 说明的json文件的class_detail数据
            class_detail_list['class_name'] = class_dir  # 类别名称
            class_detail_list['class_label'] = class_label  # 类别标签
            class_detail_list['class_eval_images'] = eval_sum  # 该类数据的测试集数目
            class_detail_list['class_trainer_images'] = trainer_sum  # 该类数据的训练集数目
            class_detail.append(class_detail_list)
            # 初始化标签列表
            train_parameters['label_dict'][str(class_label)] = class_dir
            class_label += 1

    # 初始化分类数
    train_parameters['class_dim'] = class_dim

    # 乱序
    random.shuffle(eval_list)
    with open(eval_list_path, 'a') as f:
        for eval_image in eval_list:
            f.write(eval_image)

    random.shuffle(trainer_list)
    with open(train_list_path, 'a') as f2:
        for train_image in trainer_list:
            f2.write(train_image)

    # 说明的json文件信息
    readjson = {'all_class_name': data_list_path, 'all_class_images': all_class_images, 'class_detail': class_detail}
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(train_parameters['readme_path'], 'w') as f:
        f.write(jsons)
    print('生成数据列表完成！')


class FoodsDataSet(paddle.io.Dataset):
    def __init__(self, path, im_size=None):
        super(FoodsDataSet, self).__init__()
        self.list_file = path
        if im_size is None:
            im_size = [3, 64, 64]
        self.size = im_size
        self.records_list = []
        self.parse_dataset()

    def parse_dataset(self):
        """
        处理数据集
        """
        with open(self.list_file, 'r') as f:
            for line in f.readlines():
                img_path, label = line.strip().split('\t')
                self.records_list.append([img_path, label])
        random.shuffle(self.records_list)

    def __getitem__(self, idx):
        """
         每次迭代时返回数据和对应的标签
        """
        img = self.load_img(self.records_list[idx][0])
        label = int(self.records_list[idx][1])
        return img, label

    def load_img(self, path):
        """
        从磁盘读取图片并处理
        """
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((self.size[1], self.size[2]), Image.BILINEAR)
        img = np.array(img).astype('float32')
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = img / 255  # 像素值归一化
        return img

    def __len__(self):
        return len(self.records_list)


# 初始化参数
src_path = train_parameters['src_path']
target_path = train_parameters['target_path']
train_list_path = train_parameters['train_list_path']
eval_list_path = train_parameters['eval_list_path']
batch_size = train_parameters['batch_size']
version = train_parameters['version']

# 解压原始数据到指定路径
unzip_data(src_path, target_path)

# 划分训练集与验证集，乱序，生成数据列表
with open(train_list_path, 'w') as f:
    f.seek(0)
    f.truncate()
with open(eval_list_path, 'w') as f:
    f.seek(0)
    f.truncate()

# 生成数据列表
get_data_list(target_path, train_list_path, eval_list_path)

train_set = FoodsDataSet(train_list_path, train_parameters['input_size'])
train_loader = paddle.io.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
eval_set = FoodsDataSet(eval_list_path, train_parameters['input_size'])
eval_loader = paddle.io.DataLoader(eval_set, batch_size=batch_size, shuffle=True, drop_last=True)


class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
            paddle.nn.Conv2D(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
            paddle.nn.BatchNorm2D(256),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            paddle.nn.Conv2D(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            paddle.nn.Conv2D(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
            paddle.nn.BatchNorm2D(256),
            paddle.nn.ReLU()
        )
        self.flatten = paddle.nn.Flatten()
        self.fc = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=6400, out_features=1024),
            paddle.nn.Dropout(),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=1024, out_features=256),
            paddle.nn.Dropout(),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=256, out_features=5)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


mynet = MyNet()
paddle.summary(mynet, (-1, 3, 224, 224))

model = paddle.Model(mynet)

lr = train_parameters['learning_strategy']['lr']
epochs = train_parameters['num_epochs']
log_dir = train_parameters['log_dir'] + version + '/'
save_dir = train_parameters['save_dir'] + version + '/'
final_model_path = save_dir + 'final.pdparams'

loss_fn = paddle.nn.CrossEntropyLoss()
optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
model.prepare(optimizer, loss_fn, Accuracy())
callback = paddle.callbacks.VisualDL(log_dir=log_dir)
model.fit(train_set, eval_set, epochs=epochs, batch_size=batch_size, log_freq=1000, save_dir=save_dir,
          callbacks=callback)

# 验证
mynet = MyNet()
model = paddle.Model(mynet)
model.load(final_model_path)
optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
model.prepare(optimizer, loss_fn, Accuracy())
results = model.evaluate(eval_set, batch_size=batch_size, verbose=1)
print(results)
