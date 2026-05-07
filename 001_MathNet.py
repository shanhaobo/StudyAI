import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import random
from dataclasses import dataclass

from KongMing.Utils.ConfigUtils import ApplyConfigFromKV

@dataclass
class TrainConfig:
    DatasetLen    : int   = 1000000
    EpochCnt      : int   = 1
    BatchSize     : int   = 320
    HiddenSize    : int   = 128
    LearningRate  : float = 0.1
    PrintEvery    : int   = 10

Config = TrainConfig()
Overridden = set(ApplyConfigFromKV(Config))

class Calculator:
    def __init__(self):
        self.operation_dict = {
            0: self.add,
            1: self.subtract,
            2: self.multiply,
            3: self.divide,
            4: self.power
        }

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            return float('-inf')  # 返回一个特定的值
        else:
            return a / b

    def power(self, a, b):
        return (a ** b)

    def calculate(self, a, b, idx):
        return self.operation_dict[idx](a, b)

# 定义神经网络架构
class MathNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MathNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 定义数据集
class MathDataset(Dataset):
    def __init__(self):
        self.data = torch.empty(10, 3)
        self.labels = torch.empty(10, 1)

    def __len__(self):
        return Config.DatasetLen

    def __getitem__(self, idx):
        tensor = torch.empty(3)
        # 生成范围在(0, 2^32)的随机浮点数并将其分配给张量的前两个元素
        tensor[0] = torch.FloatTensor(1).uniform_(0, 100)
        tensor[1] = torch.FloatTensor(1).uniform_(0, 100)

        idx = random.randint(0, 2)
        # 生成范围在(0, 6)的随机整数并将其分配给张量的第三个元素
        tensor[2] = idx

        label = Calculator().calculate(tensor[0], tensor[1], idx)

        return tensor, torch.Tensor(label)

############

def main():
    if Overridden:
        print("[Config] CLI overrides:", sorted(Overridden))

    dataset = MathDataset()
    dataloader = DataLoader(dataset, batch_size=Config.BatchSize, shuffle=True)

    # 创建并训练网络
    net = MathNet(3, Config.HiddenSize, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=Config.LearningRate)

    for epoch in range(Config.EpochCnt):
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.float() # 确保输入数据的数据类型为浮点型
            labels = labels.view(-1, 1).float() # 调整标签的维度并确保其数据类型为浮点型
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % Config.PrintEvery == 0:
                print(f'Epoch {epoch}, Iteration {i}, Loss {loss.item()}')


if __name__ == '__main__':
    main()
