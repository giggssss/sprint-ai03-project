import os
import torch
import torch.optim as optim
from tqdm import tqdm
from dataset import dataloader_fn, YoloDataset
from model_utils import build_model
from model import YoloLoss
from train import train as yolo_train


def train_fasterrcnn(model, args, device):
    """Training loop for Faster R-CNN"""
    train_loader = dataloader_fn(
        img_dir=os.path.join(args.data_dir, 'train_images'),
        annotation_dir=os.path.join(args.data_dir, 'train_annotations'),
        batch_size=args.batch_size, shuffle=True, device=device
    )
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    for epoch in range(args.num_epochs):
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), "fasterrcnn_model.pt")
    print("Faster R-CNN model saved.")


def train_yolo11(model, args, device):
    """Training loop for custom YOLO11"""
    train_ds = YoloDataset(
        os.path.join(args.data_dir, 'train_images'),
        os.path.join(args.data_dir, 'train_labels'),
        num_classes=args.num_classes
    )
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = YoloLoss()
    for epoch in range(args.num_epochs):
        loss = yolo_train(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, YOLO11 Loss: {loss:.4f}")
    torch.save(model.state_dict(), "yolo11_model.pt")
    print("YOLO11 model saved.")


def train_ultralytics(model, args, device):
    """Training via Ultralytics YOLO API"""
    model.train(
        data=os.path.join(args.data_dir, 'data.yaml'), epochs=args.num_epochs,
        imgsz=224, batch=args.batch_size, device=device
    )
    model.save("ultralytics_yolo_model.pt")
    print("Ultralytics YOLO training completed.")
