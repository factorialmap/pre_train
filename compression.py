import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# step1: config device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# step2: tranformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# step3: load data
data_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
data_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
loader_train = DataLoader(data_train, batch_size=64, shuffle=True)
loader_test = DataLoader(data_test, batch_size=64, shuffle=True)

# step4: check images

# obter um lote de imagens e label do loader de treino
images, labels = next(iter(loader_train))

# criar uma figura com duas subtramas
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# iterar sobre as primeiras imagens
for i in range(2):
    ax = axes[i]
    # o tensor da img tem formato (C,H,W), mas o matplolib
    # espera (H,W,C) ou (H,W)
    # tamber rmv normalização
    image = images[i].numpy().squeeze() /2 + 0.5
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Label: {labels[i]}')
    ax.axis('off')

# exibir a figura
plt.tight_layout()
plt.show()

# step5: Modelo autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128,64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh(), #Usar tanh para esclar a saida int [-1,1]
            nn.Unflatten(1,(1,28,28)) #remodelar vetor form img
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder().to(device)
criterion_ae = nn.MSELoss()
optimizer_ae = optim.Adam(autoencoder.parameters(), lr=0.001)

# step6: mdl train
for epoch in range(5):
    autoencoder.train()
    total_loss = 0
    for data, _ in loader_train:
        data = data.to(device)
        optimizer_ae.zero_grad()
        output = autoencoder(data)
        loss = criterion_ae(output, data)
        loss.backward()
        optimizer_ae.step()
        total_loss += loss.item()
    print(f"Pre-treinamento - Época {epoch+1}, Loss: {total_loss:.4f}")



# stp7: Modelo classificador usando o encoder do autoencoder
class Classifier(nn.Module):
    def __init__(self, encoder):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(64,10)
    
    def forward(self, x):
        x =self.encoder(x)
        x = self.classifier(x)
        return x

classifier = Classifier(autoencoder.encoder).to(device)
criterion_clf = nn.CrossEntropyLoss()
optimizer_clf = optim.Adam(classifier.parameters(), lr = 1e-3)

# step8: Treinamento do classificador
for epoch in range(5):
    classifier.trian()
    total_loss = 0
    for data, target in loader_train:
        data, target = data.to(device), target.to(device)
        optimizer_clf.zero_grad()
        output = classifier(data)
        loss = criterion_clf(output, target)
        loss.backward()
        optimizer_clf.step()
        total_loss += loss.item()
    print(f"Fine-tuning - Época {epoch+1}, Loss: {total_loss:.4f}")

