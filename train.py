import torch
from tqdm import tqdm

def train(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    Args:
        model: PyTorch model
        dataloader: DataLoader providing (images, targets)
        optimizer: optimizer
        criterion: loss function
        device: torch.device
    Returns:
        Average loss over the epoch
    """
    model.train()
    total_loss = 0
    for imgs, targets in tqdm(dataloader, desc="Training", leave=False):
        imgs = imgs.to(device)
        targets = targets.to(device)
        preds = model(imgs)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
