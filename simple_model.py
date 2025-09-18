import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



# step1: define dataset
x = torch.tensor([[6,2],[5,2],[1,3],[7,6]]).float()
y = torch.tensor([1,5,2,5]).float()

# step2: model specification
class Mynn(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix1 = nn.Linear(2,8, bias = False)
        self.matrix2 = nn.Linear(8,1, bias = False)
    def forward(self, x):
        x = self.matrix1(x)
        x = self.matrix2(x)
        return x.squeeze()


mdl_fit = Mynn()
optimizer = optim.SGD(mdl_fit.parameters(), lr = 0.001)
criterion = nn.MSELoss()

# step3: training loop
losses = []
for epoch in range(500):
    optimizer.zero_grad()
    loss_value = criterion(mdl_fit(x), y)
    loss_value.backward()
    optimizer.step()
    losses.append(loss_value.item())
    
y
mdl_fit(x)



sample_relu = torch.tensor([[4,6,2,-6,2,5],[1,6,2,-6,5,5]])
apply_relu = nn.ReLU()
apply_relu(sample_relu)

normal = torch.linspace(-5,5,50)
with_relu = apply_relu(normal)

plt.plot(normal.numpy(), with_relu.numpy())
plt.grid()


