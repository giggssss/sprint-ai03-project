import torch
import torchvision.transforms as T
from PIL import Image
from model import YOLO11, YoloDataset, YoloLoss


def inference(model, image_path, device, conf_threshold=0.5):
    """
    Run inference on a single image.
    Args:
        model: PyTorch detection model
        image_path (str): Path to input image
        device: torch.device
        conf_threshold (float): Confidence threshold to filter boxes
    Returns:
        List of detected boxes dict with keys 'class', 'confidence', 'bbox'
    """
    model.eval()
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    img = Image.open(image_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp)[0]  # (grid_size, grid_size, 5+num_classes)
    boxes = []
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            x, y, w, h, conf = pred[j, i, :5]
            if conf > conf_threshold:
                cls_conf, cls_idx = torch.max(pred[j, i, 5:], 0)
                boxes.append({
                    'class': int(cls_idx.item()),
                    'confidence': float(conf.item() * cls_conf.item()),
                    'bbox': [float(x), float(y), float(w), float(h)]
                })
    return boxes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLO inference")
    parser.add_argument('model_path', type=str, help='Path to saved model checkpoint')
    parser.add_argument('image_path', type=str, help='Path to image to infer')
    parser.add_argument('--device', type=str, default=None, help='device string, e.g., cuda, mps, cpu')
    parser.add_argument('--threshold', type=float, default=0.5, help='confidence threshold')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Load model architecture and weights
    model = YOLO11()
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    detections = inference(model, args.image_path, device, args.threshold)
    print(detections)
