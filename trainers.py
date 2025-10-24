import os
import torch
import torch.optim as optim
from tqdm import tqdm
from dataset import dataloader_fn, YoloDataset
from model_utils import build_model
from model import YoloLoss
from train import train as yolo_train
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics.data.augment import BaseTransform


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
    """Training via Ultralytics YOLO API with built-in augmentations"""
    
    # NMS 설정 변경
    from ultralytics.utils.ops import Profile
    Profile.time_limit = 30  # NMS 시간 제한을 30초로 증가
    
    # YOLO 학습 설정
    model.train(
        data=args.data_yaml,
        epochs=args.num_epochs,
        imgsz=640,
        batch=args.batch_size,
        device=device,
        pretrained=args.pretrained,
        workers=args.num_workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        overlap_mask=False,  # 마스크 오버랩 비활성화로 속도 향상
        max_det=300,        # 최대 검출 객체 수 제한
        iou=0.7,            # NMS IoU 임계값
        
        augment=True,  # YOLO의 내장 augmentation 활성화
        # 추가 augmentation 설정
        mosaic=1.0,    # mosaic augmentation 확률
        mixup=0.5,     # mixup augmentation 확률
        copy_paste=0.5, # copy-paste augmentation 확률
        degrees=45.0,  # 최대 회전 각도
        translate=0.2, # 최대 이동 비율
        scale=0.5,    # 스케일 범위
        shear=10.0,   # 최대 전단 각도
        perspective=0.001,  # 원근 변환
        flipud=0.5,   # 상하 뒤집기 확률
        fliplr=0.5,   # 좌우 뒤집기 확률
        hsv_h=0.015,  # HSV 색조 증강
        hsv_s=0.7,    # HSV 채도 증강
        hsv_v=0.4,    # HSV 명도 증강
    )
    
    # 모델 저장
    model.save("ultralytics_yolo_model.pt")
    print("Ultralytics YOLO training completed.")
