import json
import os
import argparse
from collections import Counter

"""
데이터셋 라벨 분포 비교 스크립트 (Train vs Valid)

이 스크립트는 학습 데이터와 검증 데이터의 라벨 분포(Fashion Style, Shot Type)를 비교 분석합니다.

사용법:
    python scripts/labeling/check_mapping.py --train train_aug_relabeled.jsonl --valid validation_relabeled.jsonl
"""

def get_stats(dataset_path):
    """데이터셋에서 Style과 Shot Type 카운트를 반환"""
    style_counts = Counter()
    shot_counts = Counter()
    total_count = 0
    
    if not os.path.exists(dataset_path):
        print(f"⚠️ Warning: File not found: {dataset_path}")
        return None, None, 0

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            total_count += 1
            try:
                item = json.loads(line)
                meta = item.get('image_metadata', {})
                
                # Fashion Style
                style = meta.get('fashion_style')
                if style:
                    style_counts[style] += 1
                
                # Shot Type
                shot = meta.get('shot_type')
                if shot:
                    shot_counts[shot] += 1
                    
            except json.JSONDecodeError:
                continue
                
    return style_counts, shot_counts, total_count

def print_comparison_table(title, train_counts, valid_counts, train_total, valid_total, keys=None):
    """비교 테이블 출력"""
    print(f"\n📊 {title} Comparison")
    print("-" * 85)
    print(f"{'Class Name':<25} | {'Train Count':<12} | {'Train %':<8} | {'Valid Count':<12} | {'Valid %':<8}")
    print("-" * 85)
    
    # 키 목록 결정 (합집합)
    if keys is None:
        keys = sorted(list(set(train_counts.keys()) | set(valid_counts.keys())))
        
    for key in keys:
        t_cnt = train_counts.get(key, 0)
        v_cnt = valid_counts.get(key, 0)
        
        t_ratio = (t_cnt / train_total * 100) if train_total > 0 else 0
        v_ratio = (v_cnt / valid_total * 100) if valid_total > 0 else 0
        
        print(f"{key:<25} | {t_cnt:<12} | {t_ratio:>6.1f}%  | {v_cnt:<12} | {v_ratio:>6.1f}%")
    print("-" * 85)

def check_mapping_stats(train_path, valid_path):
    print(f"🔍 Analyzing datasets...")
    print(f"  - Train: {train_path}")
    print(f"  - Valid: {valid_path}")
    
    # 1. Get Stats
    t_style, t_shot, t_total = get_stats(train_path)
    v_style, v_shot, v_total = get_stats(valid_path)
    
    if t_total == 0 and v_total == 0:
        print("❌ No data found.")
        return

    print(f"\n📈 Total Images")
    print(f"  - Train: {t_total}")
    print(f"  - Valid: {v_total}")
    
    # 2. Fashion Style Comparison
    # Define standard order for better readability
    style_order = ["Casual_Basic", "Street_Hip", "Sporty_Athleisure", "Chic_Modern", "Classy_Elegant"]
    # Add any extra styles found
    all_styles = set(t_style.keys()) | set(v_style.keys())
    extra_styles = sorted(list(all_styles - set(style_order)))
    final_style_order = style_order + extra_styles
    
    print_comparison_table("Fashion Style", t_style, v_style, t_total, v_total, final_style_order)
    
    # 3. Shot Type Comparison
    shot_order = ["Selfie_CloseUp", "Mirrored_Selfie", "Others_Cam"] # Standard 3 classes
    all_shots = set(t_shot.keys()) | set(v_shot.keys())
    extra_shots = sorted(list(all_shots - set(shot_order)))
    final_shot_order = shot_order + extra_shots
    
    print_comparison_table("Shot Type", t_shot, v_shot, t_total, v_total, final_shot_order)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="train_aug_relabeled.jsonl", help="Path to Train JSONL")
    parser.add_argument("--valid", type=str, default="validation_relabeled.jsonl", help="Path to Validation JSONL")
    
    args = parser.parse_args()
    
    # Handle relative paths helper
    def resolve_path(p):
        if not os.path.exists(p) and os.path.exists(os.path.join("..", "..", p)):
            return os.path.join("..", "..", p)
        return p

    train_path = resolve_path(args.train)
    valid_path = resolve_path(args.valid)
         
    check_mapping_stats(train_path, valid_path)
