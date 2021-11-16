import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor


test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=False,
    transform=ToTensor()
)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


model_path = './module/module.save'
model = NeuralNetwork()
model.load_state_dict(torch.load(model_path))
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
j = 0
r = 100
for i in range(r):
    x, y = test_data[i][0], test_data[i][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
        if predicted is actual:
            j += 1

print(f'Testing Accuracy: "{j / r}"')
