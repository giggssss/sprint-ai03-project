import argparse
import os
import glob
from torchvision import models, transforms
from torch.nn.functional import cosine_similarity
from PIL import Image, ImageDraw, ImageFont
import torch
import tqdm

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
    model = models.vit_b_16(weights="ViT_B_16_Weights.DEFAULT").to(device)
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
            for path in tqdm(image_paths, desc="Processing images"):
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

def write_results_to_file(results, output_folder="/Volumes/Macintosh SUB/Dataset/ai03-level1-project/similarity_results"):
    """
    Combine test images with their top-1 similar train images and save the results.

    Args:
        results (dict): Dictionary where keys are test image paths and values are lists of top similar train image paths.
        output_folder (str): Folder to save the combined images.
    """
    os.makedirs(output_folder, exist_ok=True)

    for test_img, similar_imgs in results.items():
        # Load test image and top-1 similar train image
        test_image = Image.open(test_img).convert("RGB")
        top1_image = Image.open(similar_imgs[0]).convert("RGB")

        # Resize images to the same height
        height = max(test_image.height, top1_image.height)
        test_image = test_image.resize((int(test_image.width * height / test_image.height), height))
        top1_image = top1_image.resize((int(top1_image.width * height / top1_image.height), height))

        # Create a new image with enough width to hold both images side by side
        combined_width = test_image.width + top1_image.width
        combined_image = Image.new("RGB", (combined_width, height + 50), "white")

        # Paste the images side by side
        combined_image.paste(test_image, (0, 50))
        combined_image.paste(top1_image, (test_image.width, 50))

        # Add text (file names) at the top
        draw = ImageDraw.Draw(combined_image)
        font = ImageFont.load_default()  # Use default font
        draw.text((10, 10), f"Test: {os.path.basename(test_img)}", fill="black", font=font)
        draw.text((test_image.width + 10, 10), f"Top1: {os.path.basename(similar_imgs[0])}", fill="black", font=font)

        # Save the combined image
        output_path = os.path.join(output_folder, os.path.basename(test_img))
        combined_image.save(output_path)
        print(f"Saved: {output_path}")


if __name__ == '__main__':
    train_images, test_images = load_data()
    print(f"훈련 이미지 수: {len(train_images)}, 테스트 이미지 수: {len(test_images)}")
    similar_images = preprocess_data(train_images, test_images)
    write_results_to_file(similar_images)
    # for test_img, sim_imgs in similar_images.items():
    #     print(f"Test Image: {test_img}")
    #     for rank, sim_img in enumerate(sim_imgs, start=1):
    #         print(f"  Top {rank}: {sim_img}")