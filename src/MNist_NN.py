import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf


class mnist_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_mnist_data(batch_size=64):
    #bring in the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

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

def load_config() -> DictConfig:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config_MNIST.yaml"
    return OmegaConf.load(config_path)


def train(cfg: DictConfig):
    train_loader: DataLoader; test_loader: DataLoader = load_mnist_data(batch_size=cfg.batch_size)

    model = mnist_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.max_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader_:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{cfg.max_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

    return accuracy, epoch + 1, model



def main(cfg: DictConfig | None = None):
    if cfg is None:
        cfg = load_config()

    full_group_name: str = "MNist NN"
    if cfg.use_wandb:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(project=cfg.wandb_project,
               name = f"mappo_{cfg.layout}_date{date_str}_seed{cfg.seed}",
               config=OmegaConf.to_container(cfg, resolve=True),
               tags = [cfg.layout, f"seed{cfg.seed}", "mappo"],
               group = full_group_name,
               notes = "config_mappo")
    
    accuracy, training_step, trained_model = train(cfg)
    
    

    if cfg.use_wandb:
        wandb.finish()

    return training_step


if __name__ == "__main__":
    main()

