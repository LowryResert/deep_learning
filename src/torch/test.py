import torch
import torch.nn as nn
import tensorwatch as tw
import torchvision


# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28 * 28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10)
#         )
#
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.linear_relu_stack(x)
#         return x


# model = NeuralNetwork()
# print(model)
model = torchvision.models.alexnet()
print("*"*50)
for name, module in model.named_children():
    print(name)
    print(module)

