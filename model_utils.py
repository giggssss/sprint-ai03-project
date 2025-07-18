from torchvision import models


def build_model(args):
    """Instantiate model based on args.model_type"""
    if args.model_type == 'fasterrcnn':
        model = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_f = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_f, args.num_classes)
    elif args.model_type == 'yolo11':
        from model import YOLO11
        model = YOLO11(num_classes=args.num_classes)
    else:
        from ultralytics import YOLO
        model = YOLO(args.yolo_weights)
    return model
