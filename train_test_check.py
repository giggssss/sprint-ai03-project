import argparse
import os
import glob
from torchvision import models, transforms
from torch.nn.functional import cosine_similarity
from PIL import Image
import torch

def load_data():
    train_imges = glob.glob("/Volumes/Macintosh SUB/Dataset/ai03-level1-project/train_images/*.png")
    test_imges = glob.glob("/Volumes/Macintosh SUB/Dataset/ai03-level1-project/test_images/*.png")
    return train_imges, test_imges

def preprocess_data(train_images, test_images):
    """
    Use pretrained ViT model to find top-10 similar train images for each test image.

    Args:
        train_images (list): List of paths to training images.
        test_images (list): List of paths to testing images.

    Returns:
        dict: A dictionary where keys are test image paths and values are lists of top-10 similar train image paths.
    """
    # Load pretrained ViT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vit_b_16(pretrained=True).to(device)
    model.eval()

    # Define image preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Extract features for a batch of images
    def extract_features(image_paths):
        features = []
        with torch.no_grad():
            for path in image_paths:
                image = Image.open(path).convert("RGB")
                input_tensor = preprocess(image).unsqueeze(0).to(device)
                feature = model(input_tensor).squeeze(0).cpu()
                features.append(feature)
        return torch.stack(features)

    # Extract features for train and test images
    print("Extracting features for training images...")
    train_features = extract_features(train_images)
    print("Extracting features for testing images...")
    test_features = extract_features(test_images)

    # Compute similarity and find top-10 matches
    print("Computing similarities...")
    results = {}
    for i, test_feature in enumerate(test_features):
        similarities = cosine_similarity(test_feature.unsqueeze(0), train_features)
        top_indices = torch.topk(similarities, k=10).indices
        results[test_images[i]] = [train_images[idx] for idx in top_indices]

    return results

if __name__ == '__main__':
    train_images, test_images = load_data()
    print(f"훈련 이미지 수: {len(train_images)}, 테스트 이미지 수: {len(test_images)}")
    similar_images = preprocess_data(train_images, test_images)
    for test_img, sim_imgs in similar_images.items():
        print(f"Test Image: {test_img}")
        for rank, sim_img in enumerate(sim_imgs, start=1):
            print(f"  Top {rank}: {sim_img}")   