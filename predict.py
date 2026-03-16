import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from models import CNN

classes = [
'airplane','automobile','bird','cat','deer',
'dog','frog','horse','ship','truck'
]

transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

model = CNN()
model.load_state_dict(torch.load("model_path"))
model.eval()

plt.figure(figsize=(10, 4))

for i in range(3):
    index = np.random.randint(0, len(test_dataset)-1)
    image, label = test_dataset[index]

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        
    plt.subplot(1, 3, i+1)
    plt.imshow(image.permute(1, 2, 0), interpolation='nearest')
    plt.title(f"Prediction: {classes[predicted]} | Real: {classes[label]}")
    plt.axis("off")

plt.show()