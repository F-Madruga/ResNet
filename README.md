# ResNet

Implementation from [https://www.youtube.com/watch?v=DkNIBBBvcPs](https://www.youtube.com/watch?v=DkNIBBBvcPs)

ResNet paper - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## Usage

```py
from resnet import ResNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def load_dataset(batch_size):
    train_dataset = torchvision.datasets.MNIST(
        root='./datasets', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(
        root='./datasets', train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader

# device configuration (prioritize GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_dataset(batch_size)
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap='gray')
plt.show()

# create ResNet
# since the image is black and white we only have 1 channel (RGB is 3)
# in this dataset we have 10 classes
model = ResNet.ResNet101(img_channels=1, num_classes=10).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print some data
        if (i + 1) % 100 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

# test and evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')
```
