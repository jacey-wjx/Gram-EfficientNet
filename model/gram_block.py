import torch
import torch.nn as nn

class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, x):
        a, b, c, d = x.size()
        feature = x.view(a, b, c * d)
        feature_t = feature.transpose(1,2)

        gram = feature.bmm(feature_t)
        a, b, c = gram.size()
        gram = gram.view(a, 1, b, c)
        return gram


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