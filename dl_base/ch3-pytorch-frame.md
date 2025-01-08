# ch3 pytorch framework

## dataset
```text
def init
def getitem  
处理单个样本，遍历dataloader时调用 
def len
```

## dataloader
### collate_fn 
```text
一个batch调用一次，处理batch，遍历dataloader时调用
```
```text
collate_fn 是 PyTorch 中 DataLoader 类的一个非常有用的参数，它允许用户自定义如何将多个数据样本组合成一个批次（batch）。在默认情况下，DataLoader 会简单地堆叠（stack）样本，但这并不总是适用于所有类型的数据，特别是当数据不仅仅是简单的张量（tensor）时。

collate_fn 是一个函数，它接受一个批次的样本列表作为输入，并输出一个处理后的批次。这个函数通常用于执行以下任务之一或多个：

处理非张量数据：如果你的数据不仅仅是张量（比如包含图像文件名、标签或其他类型的对象），collate_fn 可以用来将这些数据转换为张量或进行其他必要的预处理。
自定义批次维度：默认情况下，DataLoader 会沿着一个新的维度（通常是第0维）堆叠样本。但是，有时候你可能希望以不同的方式组合样本，比如将序列长度不同的数据填充到相同的长度，或者将不同模态的数据合并到一个批次中。
应用数据增强：在某些情况下，你可能希望在每个批次应用不同的数据增强策略。虽然这通常不是在collate_fn中完成的（因为数据增强通常应该在数据加载时就应用），但在某些特殊情况下，它可以在这里处理。
动态调整批次大小：对于某些类型的数据（如变长的序列），你可能需要动态地调整批次大小以确保所有样本都能被有效地处理。
下面是一个简单的例子，展示了如何使用collate_fn来处理变长的序列数据：

python
import torch
from torch.utils.data import DataLoader, Dataset
 
class MyDataset(Dataset):
    # 假设数据集包含不同长度的序列和对应的标签
    def __init__(self):
        self.data = [torch.randn(i) for i in range(10, 20)]  # 示例数据
        self.labels = torch.randint(0, 2, (10,))  # 示例标签
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
 
def my_collate_fn(batch):
    # batch 是一个列表，每个元素是一个 (data, label) 对
    data, labels = zip(*batch)
    # 将数据填充到相同的长度（这里简单使用0填充，实际中可能需要更复杂的处理）
    max_len = max(d.size(0) for d in data)
    padded_data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    return padded_data, torch.tensor(labels)
 
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=2, collate_fn=my_collate_fn)
 
for batch in dataloader:
    print(batch)
在这个例子中，my_collate_fn 函数将不同长度的序列数据填充到相同的长度，并返回处理后的批次数据和标签。这是处理变长序列数据时非常常见的一个需求。
```