import torch
import torch.nn as nn
import torch.optim as optim

linear = nn.Linear(3,1)
opt = optim.SGD(linear.parameters(), lr=0.1)
x = torch.Tensor([[1,1,1],[2,2,2]])
y = torch.Tensor([[1.5],[1.5]])
loss_fun = nn.MSELoss()

while True:
    print("weight(before)    ",linear.weight)
    print("grad(before)      ",linear.weight.grad)
    out = linear(x)
    print("out:              ", out)
    loss = loss_fun(out, y)
    print("loss:             ", loss)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("grad(after)       ",linear.weight.grad)
    print("weight(after)     ",linear.weight)
    print("++++++++++++++++++++++++++")