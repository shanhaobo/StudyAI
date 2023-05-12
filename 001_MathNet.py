import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import random

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
            return None
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
        return 10000
    
    def __getitem__(self, idx):
        tensor = torch.empty(3)
        # 生成范围在(0, 2^32)的随机浮点数并将其分配给张量的前两个元素
        tensor[0] = torch.FloatTensor(1).uniform_(0, 2**16)
        tensor[1] = torch.FloatTensor(1).uniform_(0, 2**16)

        idx = random.randint(0, 4)
        # 生成范围在(0, 6)的随机整数并将其分配给张量的第三个元素
        tensor[2] = idx

        return tensor, Calculator().calculate(tensor[0], tensor[1], idx)
    
############

def main():
    dataset = MathDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 创建并训练网络
    net = MathNet(3, 128, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(100):  # 进行100次迭代
        for i, (inputs, labels) in enumerate(dataloader):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f'Epoch {epoch}, Iteration {i}, Loss {loss.item()}')


if __name__ == '__main__':
    main()
