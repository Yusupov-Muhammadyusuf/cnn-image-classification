import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from models import CNN
from utils import calc_acc

train_loaders, test_loaders = get_dataloaders()

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizier = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    for images, labels in train_loaders:
        optimizier.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizier.step()
        
        acc = calc_acc(outputs, labels)

torch.save(model.state_dict(), "model_path")