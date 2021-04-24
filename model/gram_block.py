import torch
import torch.nn as nn

class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, x):
        size_g = len(x[0])
        gram_input = [[0.0 for i in range(size_g)] for j in range(size_g)]

        for i in range(size_g):
            for j in range(size_g):
                gram_input[i][j] = float(torch.sum(x[0][i].mul(x[0][j])))

        return torch.tensor(gram_input).unsqueeze(0).unsqueeze(0)


class GramBlock(nn.Module):
    def __init__(self, _in_channels=3):
        super(GramBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=_in_channels, out_channels=32, kernel_size=3, stride=1, padding=2)

        self.gram = Gram()

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    # 前馈网络过程
    def forward(self, x):
        out = self.conv1(x)
        out = self.gram(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool(out)
        return out