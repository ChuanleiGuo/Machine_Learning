'''This is the pytorch's learning tutorial'''
import torch
import numpy as np

# construct a 5 * 3 matrix, uninitialized
x = torch.Tensor(5, 3)
print(x)

# construct a randomly initialized matrix
x = torch.rand(5, 3)
print(x)
print(x.size())

# Operations

y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)

print(x[:, 1])

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# converting between numpy array and torch Tensor

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

print(torch.cuda.is_available())
