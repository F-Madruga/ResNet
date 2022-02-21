# ResNet

Implementation from [https://www.youtube.com/watch?v=DkNIBBBvcPs](https://www.youtube.com/watch?v=DkNIBBBvcPs)

ResNet paper - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## Usage

```py
from resnet import ResNet
import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import Trainer


# hyperparameters
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# data
train_dataset = torchvision.datasets.MNIST(root='./datasets', train=True, transform=transforms.ToTensor(), download=True)
val_dataset = torchvision.datasets.MNIST(root='./datasets', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# training and evaluating
model = ResNet.ResNet101(img_channels=1, num_classes=10, learning_rate=learning_rate)
trainer = Trainer(gpus=1, max_epochs=num_epochs, fast_dev_run=False)
trainer.fit(model, train_loader, val_loader)
trainer.test(model, val_loader)
```
