import torch.nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from alphanet import AlphaNet
from module.dataset import Securities
import os
import time
import numpy as np

batch_size = 100
num_epochs = 50
lr = 0.0001

path = './data/test_data.zip'
training_data = Securities(path, 'train')
training_loader = DataLoader(training_data, batch_size=batch_size)
validation_data = Securities(path, 'validation')
validation_loader = DataLoader(validation_data, batch_size=batch_size)


def draw_process(title, iters, data, label):
    plt.title(title)
    plt.xlabel("Iters")
    plt.ylabel(label)
    plt.cla()
    plt.plot(iters, data, label=label)
    plt.legend()
    plt.grid()
    dir_path = './figure/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig('{}/{}.jpg'.format(dir_path, title))


model = AlphaNet()
print(model)


def evaluate(model, eval_loader, loss_fn):
    loss_set = []
    for batch_id, data in enumerate(eval_loader):
        img_data, labels = data
        pred = model(img_data)
        valid_loss = loss_fn(pred, labels)
        # print("[evaluate] expected: {}, prediction: {}".format(labels, pred))
        loss_set.append(valid_loss.item())

    eval_loss = np.array(loss_set).mean()
    return eval_loss


def train(train_loader, eval_loader):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    train_iter_id = 0
    train_iters, train_loss = [], []
    valid_iters, valid_loss = [], []
    for epoch_id in range(1, num_epochs + 1):
        model.train()
        for batch_id, data in enumerate(train_loader):
            img_data, labels = data
            # img_data = img_data.to(torch.float32)  # tensor 需要是double类型
            # labels = labels.to(torch.float32)  # labels 需要是double类型
            pred = model(img_data)
            training_loss = loss_fn(pred, labels)
            # if torch.isnan(training_loss):
            #     print(training_loss)
            print("[training] epoch: {}, step: {}, loss: {}".format(
                epoch_id,
                batch_id + 1,
                training_loss.item()
            ))

            # 每个batch记录训练集的loss和acc
            train_iter_id += 1
            train_iters.append(train_iter_id)
            train_loss.append(training_loss.item())

            # 反向传播
            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 每个epoch记录验证集的loss和acc
        model.eval()
        loss = evaluate(model, eval_loader, loss_fn)
        print("[validation] epoch: {}, loss: {}".format(epoch_id, loss))
        valid_iters.append(epoch_id)
        valid_loss.append(loss)

    draw_process("training_loss", train_iters, train_loss, "training_loss")
    draw_process("validation_loss", valid_iters, valid_loss, "validation_loss")


print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
train(training_loader, validation_loader)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# save model, load model and testing model
model_path = './work/parameter.pkl'
torch.save(model.state_dict(), model_path)

model = AlphaNet()
model.load_state_dict(torch.load(model_path))
loss_fn = torch.nn.MSELoss()
loss = evaluate(model, validation_loader, loss_fn)
print("[testing] Loss: {}".format(loss))
