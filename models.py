import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fsc1 = nn.Linear(64 * 6 * 6, 128)
        self.fsc2 = nn.Linear(128, 10)

    def forward(self, x):
        x =  self.pool(F.relu(self.conv1(x)))
        x =  self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 6 * 6)

        x = F.relu(self.fsc1(x))
        x = self.fsc2(x)

        return x