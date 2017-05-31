# -*- coding: utf-8 -*-
''' Learning record of nn in pytorch '''

import torch
from torch.autograd import Variable

# N is batch size; D_in is input dimension
# H is Hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 10, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

loss_func = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
for t in range(500):
    y_pred = model(x)

    loss = loss_func(y_pred, y)
    print(t, loss.data[0])

    model.zero_grad()

    loss.backward()

    for param in model.parameters():
        param.data -= learning_rate * param.grad.data
