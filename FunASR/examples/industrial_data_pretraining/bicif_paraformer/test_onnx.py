import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv2_drop(x)
        return x


class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.net = Net()
        self.sr = 16000

    def cal_pow2(self, value):
        return value * 2

    def forward(self, x, wav):
        x = self.net(x)
        out = torch.log(wav) / self.cal_pow2(10)
        return out, x


model = MainModel()

torch.onnx.export(model, (torch.rand(1, 3, 224, 224), torch.rand(1, 3, 224, 224)), "sample.onnx")
