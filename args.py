import argparse


def parse_args_and_prepare():
    parser = argparse.ArgumentParser(description="Train and evaluate object detection models")
    parser.add_argument('--num_classes', type=int, default=73, help='Number of classes (including background)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='/Volumes/Macintosh SUB/Dataset/ai03-level1-project', help='Path to dataset directory')
    parser.add_argument('--model_type', type=str, choices=['fasterrcnn','yolo11','ultralytics'], default='ultralytics', help='Model type to use')
    parser.add_argument('--yolo_weights', type=str, default='yolo11n.pt', help='Path to Ultralytics YOLO weights')
    return parser.parse_args()
