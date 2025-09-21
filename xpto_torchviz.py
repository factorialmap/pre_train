import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.fc1 = nn.Linear(784,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Mynn()
writer = SummaryWriter("runs/mynn_experiment") # dir for results

# to visualize nn we need to pass a dummy input through it
dummy_input = torch.randn(1, 784) # batch size of 1, 784 features

# add the model graph to the writer
writer.add_graph(model, dummy_input)
writer.close
