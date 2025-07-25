import os
import json
import glob
import argparse

def count_files_and_check_matching(base_dir):
    # Detect dataset splits by folder pattern: *_images and corresponding *_annotations
    results = {}
    # List subdirectories in base_dir
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    image_dirs = [d for d in subdirs if d.endswith('_images')]
    for img_dir_name in image_dirs:
        split = img_dir_name[:-7]  # strip '_images'
        img_dir = os.path.join(base_dir, img_dir_name)
        ann_dir = os.path.join(base_dir, f"{split}_annotations")

        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        # Find annotation JSONs recursively
        ann_files = sorted(glob.glob(os.path.join(ann_dir, '**', '*.json'), recursive=True))

        img_basenames = set(os.path.splitext(f)[0] for f in img_files)

        # annotation에서 참조하는 이미지 파일명 추출
        ann_img_basenames = set()
        for ann_path in ann_files:
            with open(ann_path, 'r') as f:
                data = json.load(f)
                # COCO 형식: "image" 또는 "image_id" 또는 "file_name" 등에서 추출
                # 아래는 "file_name" key를 사용하는 예시
                if 'file_name' in data:
                    ann_img_basenames.add(os.path.splitext(os.path.basename(data['file_name']))[0])
                elif 'images' in data and isinstance(data['images'], list):
                    for img_info in data['images']:
                        if 'file_name' in img_info:
                            ann_img_basenames.add(os.path.splitext(os.path.basename(img_info['file_name']))[0])
                    ann_img_basenames.add(os.path.splitext(os.path.basename(data['images'][0]['file_name']))[0])
                # 필요시 다른 key도 추가

        # Determine unmatched and matched images
        unmatched_imgs = img_basenames - ann_img_basenames
        unmatched_anns = ann_img_basenames - img_basenames
        matched_imgs = img_basenames & ann_img_basenames

        results[split] = {
            'num_images': len(img_files),
            'num_annotations': len(ann_files),
            'num_matched_images': len(matched_imgs),
            'matched_images': list(matched_imgs),
            'unmatched_images': list(unmatched_imgs),
            'unmatched_annotations': list(unmatched_anns)
        }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dataset image/annotation matching.")
    parser.add_argument("--base_path", type=str, default="/Volumes/Macintosh SUB/Dataset/ai03-level1-project", help="Base dataset directory path")
    args = parser.parse_args()
    base_path = args.base_path
    stats = count_files_and_check_matching(base_path)
    for split, info in stats.items():
        # Save the results to a JSON file for each split
        output_path = os.path.join(base_path, f"{split}_stats.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"Saved stats for {split} to {output_path}")
