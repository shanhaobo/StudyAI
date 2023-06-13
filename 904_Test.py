import torch

from Models.DiffusionModel.Utils import Extract

t = torch.tensor([[1, 2], [3, 4]])

"""
out[i][j] = input[index[i][j]][j]  # if dim == 0
out[i][j] = input[i][index[i][j]]  # if dim == 1
"""
index = [[0, 0], [1, 1]]
oo0 = [[0 for _ in range(2)] for _ in range(2)]
for i in range(2):
    for j in range(2):
        oo0[i][j] = t[index[i][j]][j]

o0 = torch.gather(t, -1, torch.tensor([[0, 0], [0, 0]]))
o1 = torch.gather(t, -1, torch.tensor([[1, 1], [1, 1]]))

print(" o0:{} \n o1:{} \n oo2:{}".format(o0, o1, oo0))

ti = torch.tensor([[1, 1], [1, 1], [1, 1]])
print(ti.shape)
print(len(ti.shape))
print(len(ti.shape)- 1)
print((1,) * (len(ti.shape) + 1))
print(((1,) * (len(ti.shape) + 1)))
print(*((1,) * (len(ti.shape) + 1)))

matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)
print(matrix[0,:])
print(matrix[:,1])
