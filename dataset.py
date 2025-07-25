import os
import glob
import shutil
import random
import json
import yaml
import argparse
import torch

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as TF
from PIL import Image


class DetectionDataset(Dataset):
    def __init__(self, img_dir, ann_dir=None, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.annotations = self._load_annotations(ann_dir)
        self.image_names = list(self.annotations.keys())
        # category mapping if available
        self.categories = getattr(self, "categories", {})

    def _load_annotations(self, ann_dir):
        ann_map = {}
        if ann_dir and os.path.isdir(ann_dir):
            files = glob.glob(os.path.join(ann_dir, "**/*.json"), recursive=True)
            if not files:
                raise FileNotFoundError(f"No JSON files in {ann_dir}")
            images, anns, cats = [], [], []
            for f in files:
                part = json.load(open(f, "r"))
                images += part.get("images", [])
                anns += part.get("annotations", [])
                cats += part.get("categories", [])
            # map and remap categories
            id2name = {c["id"]: c["name"] for c in cats}
            # Preserve original order
            unique_ids = []
            for c in cats:
                cid = c["id"]
                if cid not in unique_ids:
                    unique_ids.append(cid)
            self.id2name = id2name
            self.unique_ids = unique_ids
            self.categories = {i: id2name[cid] for i, cid in enumerate(unique_ids)}
            id2idx = {cid: i for i, cid in enumerate(unique_ids)}
            id_to_file = {img["id"]: img["file_name"] for img in images}
            for ann in anns:
                fn = id_to_file.get(ann["image_id"])
                if not fn:
                    continue
                bbox = ann["bbox"]
                xmin, ymin, w, h = bbox
                box = [xmin, ymin, xmin + w, ymin + h]
                label = id2idx.get(ann.get("category_id", 0), 0)
                ann_map.setdefault(fn, {"boxes": [], "labels": []})
                ann_map[fn]["boxes"].append(box)
                ann_map[fn]["labels"].append(label)
        else:
            for path in glob.glob(os.path.join(self.img_dir, "*")):
                ann_map[os.path.basename(path)] = {"boxes": [], "labels": []}
        return ann_map

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        fn = self.image_names[idx]
        img = Image.open(os.path.join(self.img_dir, fn)).convert("RGB")
        img = TF.to_tensor(img)
        if self.transform:
            img = self.transform(img)
        ann = self.annotations[fn]
        boxes = torch.tensor(ann["boxes"], dtype=torch.float32)
        labels = torch.tensor(ann["labels"], dtype=torch.int64)
        return img, {"boxes": boxes, "labels": labels}

    def get_class_names(self):
        """
        Return the list of class names in original COCO JSON order.
        """
        return [self.id2name[cid] for cid in self.unique_ids]


def dataloader_fn(img_dir, ann_dir, batch_size=4, shuffle=True, device=None):
    transform = T.Compose([T.Resize((224, 224)), T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    ds = DetectionDataset(img_dir, ann_dir, transform)
    def collate(batch):
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs).to(device) if device else torch.stack(imgs)
        targets = [{k: v.to(device) if device else v for k, v in t.items()} for t in targets]
        return imgs, targets
    
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, pin_memory=(device and device.type=="cuda"))


# YOLO 형식 Dataset 통합
class YoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir, num_classes=73, grid_size=7, transform=None):
        """
        YOLO 형식의 이미지-라벨 디렉토리를 처리하는 Dataset
        Args:
            images_dir (str): 이미지가 저장된 디렉토리
            labels_dir (str): YOLO 형식 라벨(.txt) 파일이 저장된 디렉토리
            num_classes (int): 클래스 개수
            grid_size (int): 격자 크기
            transform (callable, optional): 이미지 변환 함수
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.grid_size = grid_size
        self.num_classes = num_classes
        # torchvision v2 ToTensor 대체 이미 transform에서 처리 가능
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.rsplit('.', 1)[0] + '.txt')
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # YOLO 타겟 생성: (grid_size, grid_size, 5 + num_classes)
        target = torch.zeros((self.grid_size, self.grid_size, 5 + self.num_classes), dtype=torch.float32)
        if os.path.isfile(label_path):
            with open(label_path) as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.strip().split())
                    i = int(x * self.grid_size)
                    j = int(y * self.grid_size)
                    target[j, i, 0:5] = torch.tensor([x, y, w, h, 1.0])
                    target[j, i, 5 + int(cls)] = 1.0
        return img, target


def convert_coco_to_yolo(img_dir, ann_dir, out_img, out_lbl):
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    ds = DetectionDataset(img_dir, ann_dir)
    for fn in ds.image_names:
        shutil.copy(os.path.join(img_dir, fn), os.path.join(out_img, fn))
        w, h = Image.open(os.path.join(img_dir, fn)).size
        lines = []
        for box, label in zip(ds.annotations[fn]["boxes"], ds.annotations[fn]["labels"]):
            xmin, ymin, xmax, ymax = box
            xc, yc = ((xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h)
            bw, bh = ((xmax - xmin) / w, (ymax - ymin) / h)
            lines.append(f"{label} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(out_lbl, fn.rsplit(".",1)[0] + ".txt"), "w") as f:
            f.write("\n".join(lines))


def prepare_yolo_structure(root, img_src, lbl_src, ann_dir=None, train_ratio=0.8, yaml_file="data.yaml"):
    for split in ("train", "val"):
        os.makedirs(os.path.join(root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, split, "labels"), exist_ok=True)
    imgs = sorted(f for f in os.listdir(img_src) if f.lower().endswith((".jpg", ".png")))
    random.shuffle(imgs)
    split = int(len(imgs)*train_ratio)
    for name in imgs[:split]:
        shutil.copy(os.path.join(img_src, name), os.path.join(root, "train/images", name))
        shutil.copy(os.path.join(lbl_src, name.rsplit(".",1)[0]+".txt"), os.path.join(root, "train/labels", name.rsplit(".",1)[0]+".txt"))
    for name in imgs[split:]:
        shutil.copy(os.path.join(img_src, name), os.path.join(root, "val/images", name))
        shutil.copy(os.path.join(lbl_src, name.rsplit(".",1)[0]+".txt"), os.path.join(root, "val/labels", name.rsplit(".",1)[0]+".txt"))
    # Load class names from annotations if provided
    if ann_dir:
        ds = DetectionDataset(img_src, ann_dir)
        names = ds.get_class_names()
    else:
        # fallback to labels directory scanning
        names = sorted({int(open(os.path.join(lbl_src, f)).readline().split()[0])
                        for f in os.listdir(lbl_src) if f.endswith(".txt")})
        names = [str(n) for n in names]
    nc = len(names)
    data = {"path": root,
            "train": os.path.join(root, "train/images"),
            "val": os.path.join(root, "val/images"),
            "nc": nc, "names": names}
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO to YOLO conversion and structure preparation")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the dataset (e.g., /Volumes/Macintosh SUB/Dataset/ai03-level1-project)")
    args = parser.parse_args()

    # example usage
    convert_coco_to_yolo(os.path.join(args.dataset_root, "train_images"), 
                         os.path.join(args.dataset_root, "train_annotations"),
                         os.path.join(args.dataset_root, "yolo_tmp/images"), 
                         os.path.join(args.dataset_root, "yolo_tmp/labels"))
    prepare_yolo_structure(os.path.join(args.dataset_root, "yolo_data"),
                           os.path.join(args.dataset_root, "yolo_tmp/images"),
                           os.path.join(args.dataset_root, "yolo_tmp/labels"),
                           ann_dir=os.path.join(args.dataset_root, "train_annotations"),
                           train_ratio=0.8, 
                           yaml_file="data.yaml")
    