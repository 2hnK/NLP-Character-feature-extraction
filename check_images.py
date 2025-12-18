import json
import os
import argparse
from pathlib import Path

def check_images(jsonl_path, image_root):
    print(f"Checking images for dataset: {jsonl_path}")
    print(f"Image root directory: {image_root}")
    
    if not os.path.exists(jsonl_path):
        print(f"Error: Dataset file not found at {jsonl_path}")
        return
    
    if not os.path.exists(image_root):
        print(f"Error: Image root directory not found at {image_root}")
        return

    missing_count = 0
    total_pairs = 0
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                
                total_pairs += 1
                item = json.loads(line)
                pair_id = item.get('pairId')
                
                if not pair_id:
                    print(f"Warning: Row {line_idx} missing pairId")
                    continue
                
                # Check User A
                user_a = item.get('userA', {})
                gender_a = user_a.get('gender', 'male') # Default fallback, though should exist
                if not check_one_user(image_root, pair_id, gender_a):
                    print(f"[MISSING] Pair: {pair_id}, UserA ({gender_a}) image not found.")
                    missing_count += 1
                
                # Check User B
                user_b = item.get('userB', {})
                gender_b = user_b.get('gender', 'female')
                if not check_one_user(image_root, pair_id, gender_b):
                    print(f"[MISSING] Pair: {pair_id}, UserB ({gender_b}) image not found.")
                    missing_count += 1
                    
    except Exception as e:
        print(f"Error processiong jsonl: {e}")
        return

    print("-" * 30)
    print(f"Total Pairs Checked: {total_pairs}")
    if missing_count == 0:
        print("SUCCESS: All expected images found!")
    else:
        print(f"WARNING: Found {missing_count} missing images.")

def check_one_user(image_root, pair_id, gender):
    # Logic matches MatchingDataset._get_image_path
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
        return False
        
    try:
        files = os.listdir(pair_dir)
    except OSError:
        return False
        
    # Search for gender keyword in filename (case-insensitive)
    for f in files:
        if gender.lower() in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg')):
            return True
            
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify local image existence for dataset")
    parser.add_argument("--dataset_jsonl", type=str, default="data/dataset.jsonl", help="Path to jsonl file")
    parser.add_argument("--image_root", type=str, default="data/images", help="Path to images directory")
    
    args = parser.parse_args()
    
    # Convert to absolute paths for clarity in output
    abs_jsonl = os.path.abspath(args.dataset_jsonl)
    abs_root = os.path.abspath(args.image_root)
    
    check_images(abs_jsonl, abs_root)
