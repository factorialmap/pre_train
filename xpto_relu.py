
# step1: load packages
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# step2: create some data
data_sample = torch.tensor([[5,3,2,-4,2,5],[1,5,3,-5,4,5]])

# check the data shape
data_sample.shape
data_sample

# step3: apply relu function
apply_relu = nn.ReLU()
apply_relu(data_sample)


data_sample2 = torch.linspace(-5,5,50)

data_with_relu = apply_relu(data_sample2)

plt.plot(data_sample2.numpy())
plt.grid()


plt.plot(data_with_relu.numpy())
plt.grid()
