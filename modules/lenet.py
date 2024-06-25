import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5), stride=(1, 1), padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            nn.AdaptiveMaxPool2d(output_size=(4, 4))  # 自适应池化，不管输入的feature map是多大，强制要求输出的feature map大小必须是output_size大小
        )

        self.classify = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        """
        :param x: 原始图像数据, [N,1,28,28]
        :return:
        """
        z = self.features(x)  # [N,1,28,28] -> [N,50,4,4]
        z = z.view(-1, 800)  # reshape
        z = self.classify(z)
        return z


if __name__ == '__main__':
    net = LeNet()
    img = torch.randn(2, 1, 28, 28)
    scores = net(img)  # [N,1,28,28] -> [N,10] 获得得到的是每个样本属于10个类别的置信度
    print(scores)
    probs = torch.softmax(scores, dim=1)  # 求解概率值
    print(probs)
