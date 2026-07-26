import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
from omegaconf import DictConfig, ListConfig, OmegaConf
from collections import deque
from typing import Any, cast


class mnist_model(torch.nn.Module):
    def __init__(self, cfg: DictConfig | None = None):
        super().__init__()
        
        if cfg:
            self.hidden_layer_1 = cfg.hidden_layer_1
            self.hidden_layer_2 = cfg.hidden_layer_2
            self.dropout_rate = cfg.dropout_rate
        else:
            self.hidden_layer_1 = 256
            self.hidden_layer_2 = 64
            self.dropout_rate = 0.2  # Default dropout rate if no config is provided
            
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, self.hidden_layer_1)
        self.fc2 = nn.Linear(self.hidden_layer_1, self.hidden_layer_2)
        self.fc3 = nn.Linear(self.hidden_layer_2, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, p=self.dropout_rate, train=self.training)
        x = torch.relu(self.fc2(x))
        x = torch.dropout(x, p=self.dropout_rate, train=self.training)
        x = self.fc3(x)
        return x

def load_mnist_data(batch_size=64, use_cuda = False):
    #bring in the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_cuda)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False , pin_memory=use_cuda)

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
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError("Expected mapping config (DictConfig) at config/config_MNIST.yaml")
    return cfg


def train(cfg: DictConfig, device: torch.device):
    
    train_loader: DataLoader
    test_loader: DataLoader
    train_loader, test_loader = load_mnist_data(batch_size=cfg.batch_size, use_cuda=(device.type == "cuda"))
    
    last_10_epoch_accuracies: deque[float] = deque(maxlen=10)

    model = mnist_model(cfg)
    model.to(device)
    print(f"Training loop on device: {device}")
 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    training_step: int = 0
    
    for epoch in range(cfg.max_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if cfg.use_wandb:
                wandb.log(
                        {
                            "training_step": training_step,
                            "loss": loss.item(),
                            "epoch": epoch + 1,
                        }
                    )
            training_step += 1

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        last_10_epoch_accuracies.append(100 * correct / total) 

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{cfg.max_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
        if cfg.use_wandb:
            wandb.log({"accuracy": accuracy, "epoch": epoch + 1, "training_step": training_step})  
        avg_last_10_epoch_accuracy = sum(last_10_epoch_accuracies) / len(last_10_epoch_accuracies)
        if cfg.use_wandb:
            wandb.log({"last_10_epoch_accuracies": avg_last_10_epoch_accuracy, "epoch": epoch + 1, "training_step": training_step})
       

    return accuracy, epoch + 1, model, avg_last_10_epoch_accuracy


def main(cfg: DictConfig | None = None):

    start_time = datetime.now()
    
    #load the config
    if cfg is None:
        cfg = load_config()

    #select the device to use for training
    print("from config: using device:", cfg.device)
    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Training Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Training Using CPU")
    
    full_group_name: str = "MNist NN"
    if cfg.use_wandb:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb_config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        wandb.init(
            project=cfg.wandb_project,
            name=f"MNIST_NN_date{date_str}_seed{cfg.seed}",
            config=wandb_config,
            tags=[f"seed{cfg.seed}", "MNIST_NN"],
            group=full_group_name,
            notes="config_MNIST_NN",
        )
        
        wandb.define_metric("training_step")
        wandb.define_metric("loss", step_metric="training_step")
        wandb.define_metric("accuracy", step_metric="training_step")
        wandb.define_metric("last_10_epoch_accuracies", step_metric="training_step")

    accuracy, training_step, trained_model, avg_last_10_epoch_accuracy = train(cfg, device)

    if cfg.use_wandb:
        wandb.finish()
    
    end_time = datetime.now()
    print("Training time:", end_time - start_time)
    
    return training_step


if __name__ == "__main__":
    main()

