import os
import zipfile
import random
import json
import paddle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from paddle.io import Dataset
import time
from module.vgg import VGGNet
from module.googlenet import GoogLeNet
from module.models import MyMobileNetV1, MyMobileNetV2, MyResNet50

'''
参数配置
'''
train_parameters = {
    "version": "GoogLeNet-v2",
    "input_size": [3, 224, 224],  # 输入图片的shape
    "class_dim": -1,  # 分类数
    "src_path": "./data/data55190/Chinese Medicine.zip",  # 原始数据集路径
    "target_path": "./data/",  # 要解压的路径
    "train_list_path": "./data/train.txt",  # train.txt路径
    "eval_list_path": "./data/eval.txt",  # eval.txt路径
    "readme_path": "./data/readme.json",  # readme.json路径
    "label_dict": {
        '0': 'baihe',
        '1': 'gouqi',
        '2': 'huaihua',
        '3': 'jinyinhua',
        '4': 'dangsen'
    },
    "num_epochs": 50,  # 训练轮数
    "train_batch_size": 32,  # 训练时每个批次的大小
    "log_freq": 20,  # 训练过程每log_freq个epoch打印一次loss和acc
    "learning_strategy": {  # 优化函数相关的配置
        "lr": 0.0001  # 超参数学习率
    },
    "checkpoints": "./work/checkpoints"  # 保存的路径
}

version = train_parameters['version']


def unzip_data(src_path, target_path):
    """
    解压原始数据集，将src_path路径下的zip包解压至target_path目录下
    """
    if not os.path.isdir(target_path + "Chinese Medicine"):
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
    data_list_path = target_path + "Chinese Medicine/"
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
    # 读取每个类别，['river', 'lawn','church','ice','desert']
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
                if class_sum % 7 is 0:  # 每8张图片取一个做验证数据
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
    readjson = {}
    readjson['all_class_name'] = data_list_path  # 文件父目录
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(train_parameters['readme_path'], 'w') as f:
        f.write(jsons)
    print('生成数据列表完成！')


'''
参数初始化
'''
src_path = train_parameters['src_path']
target_path = train_parameters['target_path']
train_list_path = train_parameters['train_list_path']
eval_list_path = train_parameters['eval_list_path']

'''
解压原始数据到指定路径
'''
unzip_data(src_path, target_path)

'''
划分训练集与验证集，乱序，生成数据列表
'''
# 每次生成数据列表前，首先清空train.txt和eval.txt
with open(train_list_path, 'w') as f:
    f.seek(0)
    f.truncate()
with open(eval_list_path, 'w') as f:
    f.seek(0)
    f.truncate()

# 生成数据列表
get_data_list(target_path, train_list_path, eval_list_path)


class dataset(Dataset):
    def __init__(self, data_path, mode='train'):
        """
        数据读取器
        :param data_path: 数据集所在路径
        :param mode: train or eval
        """
        super().__init__()
        self.data_path = data_path
        self.img_paths = []
        self.labels = []

        if mode is 'train':
            with open(os.path.join(self.data_path, "train.txt"), "r", encoding="utf-8") as f:
                self.info = f.readlines()
            for img_info in self.info:
                img_path, label = img_info.strip().split('\t')
                self.img_paths.append(img_path)
                self.labels.append(int(label))

        else:
            with open(os.path.join(self.data_path, "eval.txt"), "r", encoding="utf-8") as f:
                self.info = f.readlines()
            for img_info in self.info:
                img_path, label = img_info.strip().split('\t')
                self.img_paths.append(img_path)
                self.labels.append(int(label))

    def __getitem__(self, index):
        """
        获取一组数据
        :param index: 文件索引号
        :return:
        """
        # 第一步打开图像文件并获取label值
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        img = np.array(img).astype('float32')
        img = img.transpose((2, 0, 1)) / 255
        label = self.labels[index]
        label = np.array([label], dtype="int64")
        return img, label

    def print_sample(self, index: int = 0):
        print("文件名", self.img_paths[index], "\t标签值", self.labels[index])

    def __len__(self):
        return len(self.img_paths)


batch_size = train_parameters["train_batch_size"]

# 训练数据加载
train_dataset = dataset('./data', mode='train')
train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 测试数据加载
eval_dataset = dataset('./data', mode='eval')
eval_loader = paddle.io.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)


def draw_process(version, title, color, iters, data, label):
    plt.title(title, fontsize=24)
    plt.xlabel("Iters", fontsize=20)
    plt.ylabel(label, fontsize=24)
    plt.cla()
    plt.plot(iters, data, color=color, label=label)
    plt.legend()
    plt.grid()
    dir_path = './figure/{}'.format(version)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig('{}/{}.jpg'.format(dir_path, title))


# model = VGGNet()
model = GoogLeNet(3, 5)
# model = MyMobileNetV1(num_classes=5)
# model = MyMobileNetV2(num_classes=5)
# model = MyResNet50(num_classes=5)

FLOPs = paddle.flops(model, [1, 3, 224, 224], print_detail=True)
print(FLOPs)


def evaluate(model, eval_loader):
    acc_set, loss_set = [], []
    for batch_id, data in enumerate(eval_loader()):
        imgs, labels = data
        predicts, acc = model(imgs, labels)
        loss = cross_entropy(predicts, labels)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        loss_set.append(float(avg_loss.numpy()))

    eval_loss = np.array(loss_set).mean()
    eval_acc = np.array(acc_set).mean()
    return eval_loss, eval_acc


cross_entropy = paddle.nn.CrossEntropyLoss()
optimizer = paddle.optimizer.Adam(learning_rate=train_parameters['learning_strategy']['lr'],
                                  parameters=model.parameters())

log_freq = train_parameters['log_freq']
# 训练集 acc 和 loss 变化, 以 batch 为单位，一共 epoch_num * batch_num 个
train_iters, train_loss, train_acc = [], [], []
# 验证集 acc 和 loss 变化, 以 epoch 为单位，一共 epoch_num 个
iters, total_loss, total_acc = [], [], []
train_iter_id = 0

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
for epoch_id in range(1, train_parameters['num_epochs'] + 1):
    model.train()
    for batch_id, data in enumerate(train_loader()):
        img, labels = data
        predicts, acc = model(img, labels)
        loss = cross_entropy(predicts, labels)
        avg_loss = paddle.mean(loss)
        if batch_id % log_freq is 0:
            print(
                "[training] epoch: {}, step: {}, loss: {}, acc: {}".format(epoch_id, batch_id + 1, avg_loss.numpy()[0],
                                                                           acc.numpy()[0]))
        # 每个batch记录训练集的loss和acc
        train_iter_id += 1
        train_iters.append(train_iter_id)
        train_loss.append(avg_loss.numpy()[0])
        train_acc.append(acc.numpy()[0])
        # 反向传播
        avg_loss.backward()
        optimizer.step()
        optimizer.clear_grad()

    # 每个epoch记录验证集的loss和acc
    model.eval()
    loss, acc = evaluate(model, eval_loader)
    print("[validation] epoch: {}, loss: {}, acc: {}".format(epoch_id, loss, acc))
    iters.append(epoch_id)
    total_loss.append(loss)
    total_acc.append(acc)

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
model_parameter_path = train_parameters["checkpoints"] + "/" + version + "/" + "save_dir_final.pdparams"
paddle.save(model.state_dict(), model_parameter_path)

draw_process(version, "trainning_loss", "red", train_iters, train_loss, "trainning loss")
draw_process(version, "trainning_acc", "green", train_iters, train_acc, "trainning acc")
draw_process(version, "validation_loss", "red", iters, total_loss, "validation loss")
draw_process(version, "validation_acc", "green", iters, total_acc, "validation acc")

# valid_model = VGGNet()
valid_model = GoogLeNet(3, 5)
# valid_model = MyMobileNetV1(num_classes=5)
# valid_model = MyMobileNetV2(num_classes=5)
# valid_model = MyResNet50(num_classes=5)
print("[validation set] evaluate start...")
param_dict = paddle.load(model_parameter_path)
valid_model.load_dict(param_dict)
valid_model.eval()
valid_loss, valid_acc = evaluate(valid_model, eval_loader)
print("[validation set] loss: {}, acc: {}".format(valid_loss, valid_acc))


def unzip_infer_data(src_path, target_path):
    """
    解压预测数据集
    """
    if not os.path.isdir(target_path + "Chinese Medicine Infer"):
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()


def load_image(img_path):
    """
    预测图片预处理
    """
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1)) / 255  # HWC to CHW 及归一化
    return img


infer_src_path = './data/data55194/Chinese Medicine Infer.zip'
infer_dst_path = './data/'
unzip_infer_data(infer_src_path, infer_dst_path)


label_dict = train_parameters['label_dict']

model__state_dict = paddle.load('work/checkpoints/{}/save_dir_final.pdparams'.format(version))
# model_predict = VGGNet()
model_predict = GoogLeNet(3, 5)
# model_predict = MyMobileNetV1(num_classes=5)
# model_predict = MyMobileNetV2(num_classes=5)
# model_predict = MyResNet50(num_classes=5)
model_predict.set_state_dict(model__state_dict)
model_predict.eval()

infer_imgs_path = os.listdir(infer_dst_path + "Chinese Medicine Infer")
print(infer_imgs_path)
for infer_img_path in infer_imgs_path:
    infer_img = load_image(infer_dst_path + "Chinese Medicine Infer/" + infer_img_path)
    infer_img = infer_img[np.newaxis, :, :, :]  # reshape(-1,3,224,224)
    infer_img = paddle.to_tensor(infer_img)
    result = model_predict(infer_img)
    lab = np.argmax(result.numpy())
    print("样本: {},被预测为:{}".format(infer_img_path, label_dict[str(lab)]))
