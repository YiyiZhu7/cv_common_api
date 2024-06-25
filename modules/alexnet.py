import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, device1, device2):
        super(AlexNet, self).__init__()
        self.device1 = device1
        self.device2 = device2

        self.feature11 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(11, 11), stride=(4, 4), padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=10),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.LocalResponseNorm(size=30),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        ).to(self.device1)
        self.feature21 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(11, 11), stride=(4, 4), padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=10),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.LocalResponseNorm(size=30),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        ).to(self.device2)
        self.feature12 = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        ).to(self.device1)
        self.feature22 = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ).to(self.device2)
        self.classify = nn.Sequential(
            nn.Linear(6 * 6 * 128 * 2, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000)
        ).to(self.device1)

    def forward(self, x):
        x1 = x.to(self.device1)  # 将x复制到设备1
        x2 = x.to(self.device2)  # 将x复制到设备2

        # 第一部分特征提取
        oz1 = self.feature11(x1)
        oz2 = self.feature21(x2)

        # 两个通道的特征合并
        z1 = torch.concat([oz1, oz2.to(self.device1)], dim=1)
        z2 = torch.concat([oz1.to(self.device2), oz2], dim=1)

        # 第二部分特征提取
        z1 = self.feature12(z1)
        z2 = self.feature22(z2)

        # 两个通道的特征合并
        z = torch.concat([z1, z2.to(self.device1)], dim=1)
        z = z.view(-1, 6 * 6 * 128 * 2)

        # 决策输出
        z = self.classify(z)

        return z


if __name__ == '__main__':
    device1 = torch.device("cpu")
    device2 = torch.device("cpu")
    net = AlexNet(device1, device2)
    img = torch.randn(2, 3, 224, 224)
    scores = net(img)
    print(scores)
    probs = torch.softmax(scores, dim=1)  # 求解概率值
    print(probs)

    # 参考pytorch中的默认实现
    from torchvision import models

    net = models.alexnet()
    print(net)
