import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt


#bring in the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def show_two_random_mnist_images(dataset):
    # pick 2 random indices
    idx = torch.randint(0, len(dataset), (2,))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for ax, i in zip(axes, idx):
        img, label = dataset[i]          # img shape: [1, 28, 28]
        img = img.squeeze(0)             # -> [28, 28]

        # If normalized with mean=0.5, std=0.5, convert back to [0,1]
        img = img * 0.5 + 0.5

        ax.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"Label: {label}")

        # Show pixel grid (28x28)
        ax.set_xticks(range(28))
        ax.set_yticks(range(28))
        ax.set_xticks([x - 0.5 for x in range(29)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(29)], minor=True)
        ax.grid(which="minor", color="lightgray", linewidth=0.5)

    plt.tight_layout()
    plt.show()

# call it
show_two_random_mnist_images(train_dataset)