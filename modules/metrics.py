import torch
import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    @torch.no_grad()
    def forward(self, scores, target) -> (int, torch.Tensor):
        """
        准确率计算方法
          N: 表示样本批次大小；
          C: 表示类别数目
        :param scores: 模型预测置信度对象 [N,C] float类型
        :param target: 样本实际标签类别对象 [N] long类型，内部就是[0,C)的索引id
        :return: (N,准确率值)
        """
        # 获取预测的标签值
        pred_indexes = torch.argmax(scores, dim=1)
        pred_indexes = pred_indexes.to(target.device, dtype=target.dtype)
        # 两者进行比较
        corr = (pred_indexes == target).to(dtype=torch.float)
        acc = torch.mean(corr)
        return corr.shape[0], acc
