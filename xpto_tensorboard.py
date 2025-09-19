import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# step1: setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# step2 : define hyperparameters
input_size = 784
hidden_size =  500
num_classes = 10
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# step3: load dataset
train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=transforms.ToTensor())

# step4: data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# step5: define model
class Mynn(nn.Module):
    def __initr__(self, input_size, hidden_size, num_classes):
        super(Mynn, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x    # no softmax at the end because CrossEntropyLoss applies it

# step6: model, loss and optimizer 
mdl_fit = Mynn(input_size, hidden_size,num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mdl_fit.parameters(), lr=learning_rate)

# step7: setup tensorboard
writer = SummaryWriter(f'runs/mnist_experiment_lr{learning_rate}_bs{batch_size}')

# -- Log sample images to TensorBoard --
images, _ = next(iter(train_loader))
img_grid = torchvision.utils.make_grid(images[:3]) # Take the first 3 images
writer.add_image('three_mnist_images', img_grid)

# -- Log model graph to TensorBoard --
images_flat = images.reshape(-1, 28*28)
writer.add_graph(model, images_flat.to(device))

# step8: training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            # Log loss to TensorBoard
            writer.add_scalar('training loss', loss.item(), epoch * n_total_steps + i)

writer.close()
print("Finished Training. Check TensorBoard by running: tensorboard --logdir=runs")
