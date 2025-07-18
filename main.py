from torchvision import models
from dataset import DetectionDataset, dataloader_fn

import os
import warnings
warnings.filterwarnings("ignore", ".*NMS time limit.*", category=UserWarning)
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from args import parse_args_and_prepare
from device import get_device
from model_utils import build_model
from trainers import train_fasterrcnn, train_yolo11, train_ultralytics


def main(args):
    # Build model and select device
    model = build_model(args)
    device = get_device()
    model.to(device)

    # Dispatch training
    if args.model_type == 'fasterrcnn':
        train_fasterrcnn(model, args, device)
    elif args.model_type == 'yolo11':
        train_yolo11(model, args, device)
    else:
        train_ultralytics(model, args, device)

if __name__ == "__main__":
    args = parse_args_and_prepare()
    main(args)