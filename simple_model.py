import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# step1: define dataset
x = torch.tensor([[6,2],[5,2],[1,3],[7,6]]).float()
y = torch.tensor([1,5,2,5]).float()

# check independent variable (features)
x.shape
x

# check dependent variable (labels)
y.shape
y

# step2: model specification
class Mynn(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix1 = nn.Linear(2,80)
        self.matrix2 = nn.Linear(80,80)
        self.matrix3 = nn.Linear(80,1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.matrix1(x))
        x = self.relu(self.matrix2(x))
        x = self.matrix3(x)
        return x.squeeze()

mdl_fit = Mynn()
optimizer = optim.AdamW(mdl_fit.parameters(), lr = 0.001)
criterion = nn.MSELoss()

# step3: setup tensorboard
writer = SummaryWriter("runs/mynn_experiment") # dir for results
writer.add_graph(mdl_fit, x)

# step4: training loop
losses = []
for epoch in range(3000):
    optimizer.zero_grad()
    loss_value = criterion(mdl_fit(x), y)
    loss_value.backward()
    optimizer.step()
    losses.append(loss_value.item())
    if epoch % 100 ==0:
        writer.add_scalar('Training Loss', loss_value.item(), epoch)

writer.close()

# step5: check results on the model training
y   # ground truth
mdl_fit(x) # predictions

# step6: visualize the loss
plt.plot(losses)
plt.title("Training Loss over epochs")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
plt.clf()


# The line below is a "magic command" for Jupyter Notebooks/IPython.
# It launches TensorBoard directly in the notebook's output.
# It will cause a SyntaxError if you run this as a standard .py script.
# %tensorboard --logdir=runs
