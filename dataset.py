import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as TF
from PIL import Image
import json
import glob

class DetectionDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        """
        Args:
            img_dir (str): Directory with all the images.
            annotation_dir (str): Directory with COCO format annotation JSON files  .
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_dir = img_dir
        self.transform = transform
        # Build annotations mapping; handle missing annotation_dir
        annotations = {}
        if annotation_dir and os.path.isdir(annotation_dir):
            json_files = glob.glob(os.path.join(annotation_dir, "**/*.json"), recursive=True)
            if not json_files:
                raise FileNotFoundError(f"No JSON annotation files found in {annotation_dir}")
            # Aggregate all COCO data
            all_images = []
            all_anns = []
            all_categories = []
            for jf in json_files:
                with open(jf, 'r') as f:
                    coco_part = json.load(f)
                all_images.extend(coco_part.get('images', []))
                all_anns.extend(coco_part.get('annotations', []))
                all_categories.extend(coco_part.get('categories', []))
            # Map category IDs to labels
            self.categories = {cat['id']: cat['name'] for cat in all_categories}
            # Remap category IDs to 0-based indices
            self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted(set(cat['id'] for cat in all_categories)))}
            self.categories = {self.cat_id_to_idx[cat_id]: name for cat_id, name in self.categories.items()}

            # Ensure all annotations have valid image IDs
            for ann in all_anns:
                if ann['image_id'] not in {img['id'] for img in all_images}:
                    raise ValueError(f"Invalid image ID {ann['image_id']} in annotation")
            # Map image IDs to file names
            id2file = {img['id']: img['file_name'] for img in all_images}
            # Group annotations by file name across all JSONs
            for ann in all_anns:
                file_name = id2file.get(ann['image_id'])
                if file_name is None:
                    continue
                # Convert bbox from [x,y,width,height] to [xmin,ymin,xmax,ymax]
                x, y, w, h = ann.get('bbox', [0, 0, 0, 0])
                bbox = [x, y, x + w, y + h]
                orig_category_id = ann.get('category_id', 0)
                category_id = self.cat_id_to_idx.get(orig_category_id, 0)
                if file_name not in annotations:
                    annotations[file_name] = {'boxes': [], 'labels': []}
                annotations[file_name]['boxes'].append(bbox)
                annotations[file_name]['labels'].append(category_id)
        else:
            # No annotations: load images only
            img_paths = glob.glob(os.path.join(self.img_dir, '*'))
            for img_path in img_paths:
                filename = os.path.basename(img_path)
                annotations[filename] = {'boxes': [], 'labels': []}
        self.annotations = annotations
        self.image_names = list(annotations.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        pil_image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")  # Load and convert to RGB
        # Convert PIL Image to torch Tensor
        image = TF.to_tensor(pil_image)
        # Apply additional transforms if provided (e.g., normalization)
        if self.transform:
            image = self.transform(image)
        ann = self.annotations[img_name]
        boxes = torch.tensor(ann['boxes'], dtype=torch.float32)
        labels = torch.tensor(ann['labels'], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        return image, target

# Note: Dummy load_annotations removed. Use COCO JSON via annotation_file in DetectionDataset.

def dataloader_fn(img_dir, annotation_dir, batch_size=4, shuffle=True, device=None):
    """
    Create DataLoader for COCO dataset.
    Args:
        img_dir (str): Directory with all the images.
        annotation_dir (str): Directory with COCO format annotation JSON files.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        device (torch.device, optional): Device to move data to.
    """
    # Data transforms using v2 APIs
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    # Dataset and DataLoader
    dataset = DetectionDataset(img_dir, annotation_dir, transform)
    # Custom collate_fn to move tensors to device
    if device is not None:
        def collate_fn(batch):
            images, targets = zip(*batch)
            images = torch.stack(list(images), dim=0).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            return images, targets
        pin_memory = (device.type == 'cuda')
    else:
        collate_fn = lambda x: tuple(zip(*x))
        pin_memory = False
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, pin_memory=pin_memory
    )
    return dataloader