from torchvision import models
from dataset import DetectionDataset, dataloader_fn

import os
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim


def parse_args_and_prepare():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN for object detection")
    parser.add_argument('--num_classes', type=int, default=73, help='Number of classes (including background)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='/Users/seokminyun/Documents/GitHub/Datas/ai03-level1-project', help='Path to dataset directory')

    args = parser.parse_args()
    return args

def main(args):
    # Model: Pretrained Faster R-CNN
    model = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, args.num_classes)

    train_loader = dataloader_fn(
        img_dir=os.path.join(args.data_dir, 'train_images'),
        annotation_dir=os.path.join(args.data_dir, 'train_annotations'),
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = dataloader_fn(
        img_dir=os.path.join(args.data_dir, 'test_images'),
        annotation_dir=None,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Select device: CUDA, MPS (Apple Silicon), or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)

    # Training loop
    model.train()
    for epoch in range(args.num_epochs):
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            images = torch.stack(list(img.to(device) for img in images), dim=0)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {losses.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "detection_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    args = parse_args_and_prepare()
    main(args)