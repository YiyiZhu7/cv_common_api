import numpy as np
import torch


def t1():
    v1 = torch.rand(2, 3, 4)
    print(v1.shape)
    print(v1)
    print("=" * 100)

    py_v2 = [[1.0, 1.320], [2.3, 3.2], [6.2, 3.2]]
    v2 = torch.tensor(py_v2)
    print(v2.shape)
    print(v2)
    print("=" * 100)
    py_v2[0][1] = 100.0
    print(py_v2)
    print(v2)
    print("=" * 100)

    np_v3 = np.random.randint(10, size=(2, 3))
    v3 = torch.from_numpy(np_v3)  # 创建的tensor对象和入参numpy对象共用同一个内存地址
    print(np_v3.shape)
    print(v3.shape)
    print(np_v3)
    print(v3)
    print("=" * 100)
    np_v3[0][1] = 100.0
    v3[1][1] = -100
    print(np_v3)
    print(v3)
    print("=" * 100)


def t2():
    v1 = torch.rand(2, 3)
    print(v1)
    np_v1 = v1.numpy()  # v1必须在cpu上、v1不涉及到梯度的计算
    print(np_v1)
    np_v1 = v1.detach().cpu().numpy()
    print(np_v1)

    v2 = torch.rand(1)[0]
    print(v2)
    print(v2.shape)
    np_v2 = v2.detach().cpu().numpy()
    print(type(np_v2), np_v2)
    py_v2 = v2.item()  # 仅支持v2是一个标量的tensor
    print(type(py_v2), py_v2)


if __name__ == '__main__':
    t2()
