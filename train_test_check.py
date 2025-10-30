import argparse
import os
import glob
import pandas as pd
import re
import json
from tqdm import tqdm

def check_annotation_consistency(annotation_root_dir, output_csv_path="inconsistency_report.csv"):
    """
    Checks for inconsistencies between parent folder names and their contents in train_annotations.
    Saves a report of inconsistencies to a CSV file.

    Args:
        annotation_root_dir (str): Path to the train_annotations directory.
        output_csv_path (str): Path to save the inconsistency report CSV.
    """
    inconsistencies = []

    for parent_dir_name in os.listdir(annotation_root_dir):
        parent_dir_path = os.path.join(annotation_root_dir, parent_dir_name)
        if not os.path.isdir(parent_dir_path) or not parent_dir_name.endswith("_json"):
            continue

        # Extract obj items from parent directory name (e.g., K-001900-010224-016551-031705_json -> 001900, 010224, ...)
        match = re.match(r"K-((\d{6}-)*\d{6})_json", parent_dir_name)
        if not match:
            inconsistencies.append({
                "Parent Folder": parent_dir_name,
                "Internal Item": "N/A",
                "Inconsistency Type": "Invalid Parent Folder Name Format",
                "Details": "Parent folder name does not match expected K-obj1-obj2-...-objN_json format"
            })
            continue
        
        obj_items = match.group(1).split('-')
        internal_items = os.listdir(parent_dir_path)

        for obj_item in obj_items:
            found_in_internal = False
            for internal_item in internal_items:
                if obj_item in internal_item:
                    found_in_internal = True
                    break
            
            if not found_in_internal:
                inconsistencies.append({
                    "Parent Folder": parent_dir_name,
                    "Internal Item": "N/A", # Specific internal item not found
                    "Inconsistency Type": "Missing Obj in Internal Item",
                    "Details": f"Obj item '{obj_item}' not found in any internal file/folder names"
                })

    if inconsistencies:
        df = pd.DataFrame(inconsistencies)
        df.to_csv(output_csv_path, index=False)
        print(f"Inconsistency report saved to {output_csv_path}")
    else:
        print("No inconsistencies found.")

def create_annotated_image_list_csv(annotation_root_dir, output_csv_path="annotated_train_images.csv"):
    """
    Creates a CSV file listing all image files that have corresponding annotations.

    Args:
        annotation_root_dir (str): Path to the train_annotations directory.
        output_csv_path (str): Path to save the CSV file.
    """
    annotated_image_names = set()
    for parent_dir_name in tqdm(os.listdir(annotation_root_dir), desc="Processing annotation folders"):
        parent_dir_path = os.path.join(annotation_root_dir, parent_dir_name)
        if not os.path.isdir(parent_dir_path) or not parent_dir_name.endswith("_json"):
            continue
        
        # Assuming annotation files are directly inside these _json folders
        json_files = glob.glob(os.path.join(parent_dir_path, "*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    for image_info in data.get('images', []):
                        annotated_image_names.add(image_info['file_name'])
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {json_file}")
                continue

    if annotated_image_names:
        df = pd.DataFrame(list(annotated_image_names), columns=["Image_Name"])
        df.to_csv(output_csv_path, index=False)
        print(f"Annotated image list saved to {output_csv_path}")
    else:
        print("No annotated images found.")

def find_images_by_inconsistency_report(train_images_dir, inconsistency_report_csv_path="inconsistency_report.csv", output_csv_path="matched_inconsistent_images.csv"):
    """
    Finds images in train_images_dir that start with 'Parent Folder' names from the inconsistency report.
    Saves the list of matched image paths to a CSV file.

    Args:
        train_images_dir (str): Path to the train_images directory.
        inconsistency_report_csv_path (str): Path to the inconsistency report CSV file.
        output_csv_path (str): Path to save the matched image list CSV.
    """
    if not os.path.exists(inconsistency_report_csv_path):
        print(f"Error: Inconsistency report CSV not found at {inconsistency_report_csv_path}")
        return

    inconsistency_df = pd.read_csv(inconsistency_report_csv_path)
    # Get unique parent folder names from the report
    parent_folder_names = inconsistency_df["Parent Folder"].unique().tolist()

    matched_images = []
    all_train_images = os.listdir(train_images_dir)

    print(f"Searching for images in {train_images_dir} based on inconsistency report...")
    for parent_name in tqdm(parent_folder_names, desc="Matching inconsistent parent folders"):
        # Extract the base name for comparison (e.g., K-001900-010224-016551-031705_json -> K-001900-010224-016551-031705)
        base_parent_name = parent_name.rsplit('_json', 1)[0]
        
        for train_img_file in all_train_images:
            base_train_img_file = os.path.splitext(train_img_file)[0]
            if base_train_img_file.startswith(base_parent_name):
                matched_images.append(os.path.join(train_images_dir, train_img_file))

    if matched_images:
        df = pd.DataFrame(list(set(matched_images)), columns=["Image_Path"])
        df.to_csv(output_csv_path, index=False)
        print(f"Matched inconsistent train image list saved to {output_csv_path}")
    else:
        print("No matching inconsistent train images found.")
    
    print(f"Total matched images: {len(matched_images)}")


if __name__ == '__main__':
    # Check annotation consistency
    annotation_dir = "/Volumes/Macintosh SUB/Dataset/ai03-level1-project/train_annotations"
    check_annotation_consistency(annotation_dir, "inconsistency_report.csv")

    # Create CSV of annotated train images
    create_annotated_image_list_csv(annotation_dir, "annotated_train_images.csv")

    # Find images in train_images that start with names from the inconsistency report
    train_images_dir = "/Volumes/Macintosh SUB/Dataset/ai03-level1-project/train_images"
    find_images_by_inconsistency_report(train_images_dir, "inconsistency_report.csv", "matched_inconsistent_images.csv")

