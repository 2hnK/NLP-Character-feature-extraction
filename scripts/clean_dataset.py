
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def get_image_path(image_root, pair_id, gender):
    """
    Finds the image path given pairId and gender.
    Replicates logic from MatchingDataset.
    """
    pair_dir = os.path.join(image_root, pair_id)
    
    # Handle zero-padding mismatch
    if not os.path.exists(pair_dir):
        try:
            parts = pair_id.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                prefix, num = parts
                alt_pair_id = f"{prefix}_{int(num):03d}"
                alt_pair_dir = os.path.join(image_root, alt_pair_id)
                if os.path.exists(alt_pair_dir):
                    pair_dir = alt_pair_dir
        except Exception:
            pass
            
    if not os.path.exists(pair_dir):
        return None
        
    try:
        files = os.listdir(pair_dir)
    except OSError:
        return None
        
    target_files = [f for f in files if gender.lower() in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not target_files:
        return None
        
    return os.path.join(pair_dir, target_files[0])

def clean_dataset(jsonl_path, image_root, output_path):
    print(f"Loading data from {jsonl_path}...")
    valid_data = []
    removed_count = 0
    total_count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    total_count = len(lines)
    print(f"Total entries: {total_count}")
    print(f"Checking images in {image_root}...")
    
    for line in tqdm(lines):
        if not line.strip():
            continue
            
        item = json.loads(line)
        pair_id = item['pairId']
        user_a = item['userA']
        user_b = item['userB']
        
        path_a = get_image_path(image_root, pair_id, user_a.get('gender', 'male'))
        path_b = get_image_path(image_root, pair_id, user_b.get('gender', 'female'))
        
        if path_a and path_b:
            valid_data.append(item)
        else:
            removed_count += 1
            
    print(f"Cleaned dataset. Removed {removed_count} entries.")
    print(f"Valid entries: {len(valid_data)}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Saved cleaned dataset to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="/home/sagemaker-user/data/dataset.jsonl")
    parser.add_argument("--image_root", type=str, default="/home/sagemaker-user/data/images")
    parser.add_argument("--output_path", type=str, default="/home/sagemaker-user/data/dataset_cleaned.jsonl")
    
    args = parser.parse_args()
    
    clean_dataset(args.jsonl_path, args.image_root, args.output_path)
