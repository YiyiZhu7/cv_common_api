import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class NumpyDataset(Dataset):

    def __init__(self, x, y):
        super(NumpyDataset, self).__init__()
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        # 返回index对应的样本数据
        return self.x[index], self.y[index]

    def __len__(self):
        # 返回当前数据集的样本条数
        return len(self.x)


def build_dataloader(X, Y, test_size, batch_size):
    # 1. 加载数据 + 数据特征工程
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=test_size, random_state=24)
    print(f"训练数据shape:{train_x.shape} - {train_y.shape}")
    print(f"训练数据shape:{test_x.shape} - {test_y.shape}")

    # 2. 构建Dataset对象
    train_dataset = NumpyDataset(x=train_x, y=train_y)
    test_dataset = NumpyDataset(x=test_x, y=test_y)

    # 3. 构建数据遍历器
    # 将dataset里面的数据一条一条的拿出来，然后合并到一起形成一个批次的数据集，并返回
    train_dataloader = DataLoader(
        dataset=train_dataset,  # 给定数据集对象，要求必须有__getitem__方法
        batch_size=batch_size,  # 批次大小
        shuffle=True,  # 在从dataset中提取数据的时候，是否需要打乱顺序
        num_workers=0,  # 数据加载形成batch的过程是否需要多线程，0表示直接在当前主线程中执行
        collate_fn=None,  # 给定如何将n条数据合并成批次数据返回，默认情况不用调整
        prefetch_factor=2  # 当num_workers为0的时候，必须为默认值；其它情况给定的是预加载的样本数目，一般情况设置为batch_size * num_workers

    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=0,
        collate_fn=None
    )
    return train_dataloader, test_dataloader, test_x, test_y
