"""
Dataset Review Script for dataset.jsonl
"""
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

def analyze_dataset(filepath):
    print(f"=== Dataset Comprehensive Review ===\n")
    print(f"File: {filepath}\n")
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # 1. Basic Stats
    print("--- 1. 기본 통계 ---")
    print(f"총 데이터 건수: {len(data)}")
    
    pair_types = Counter(d.get('pairType') for d in data)
    print(f"Positive: {pair_types.get('positive', 0)}")
    print(f"Negative: {pair_types.get('negative', 0)}")
    
    labels = Counter(d.get('label') for d in data)
    print(f"Label 분포: {dict(labels)}")
    
    # 2. Schema Check
    print("\n--- 2. 스키마 검증 ---")
    sample = data[0]
    print(f"Top-level 키: {list(sample.keys())}")
    if 'userA' in sample:
        print(f"userA 키: {list(sample['userA'].keys())}")
    if 'userB' in sample:
        print(f"userB 키: {list(sample['userB'].keys())}")
    
    # 3. Image Paths Check
    print("\n--- 3. 이미지 경로 검증 ---")
    total_images_a = 0
    total_images_b = 0
    empty_image_users = 0
    s3_paths = 0
    local_paths = 0
    
    for d in data:
        ua = d.get('userA', {})
        ub = d.get('userB', {})
        
        imgs_a = ua.get('imagePaths', [])
        imgs_b = ub.get('imagePaths', [])
        
        total_images_a += len(imgs_a)
        total_images_b += len(imgs_b)
        
        if len(imgs_a) == 0 or len(imgs_b) == 0:
            empty_image_users += 1
        
        for img in imgs_a + imgs_b:
            if 's3.amazonaws.com' in img or 's3://' in img:
                s3_paths += 1
            else:
                local_paths += 1
    
    print(f"UserA 총 이미지 수: {total_images_a}")
    print(f"UserB 총 이미지 수: {total_images_b}")
    print(f"이미지 없는 유저 포함 쌍 수: {empty_image_users}")
    print(f"S3 경로 수: {s3_paths}")
    print(f"로컬 경로 수: {local_paths}")
    
    # 4. Gender Distribution
    print("\n--- 4. 성별 분포 ---")
    gender_a = Counter(d.get('userA', {}).get('gender') for d in data)
    gender_b = Counter(d.get('userB', {}).get('gender') for d in data)
    print(f"UserA 성별: {dict(gender_a)}")
    print(f"UserB 성별: {dict(gender_b)}")
    
    # 5. Interests Check
    print("\n--- 5. 관심사 통계 ---")
    all_interests = []
    for d in data:
        all_interests.extend(d.get('userA', {}).get('interests', []))
        all_interests.extend(d.get('userB', {}).get('interests', []))
    interest_counts = Counter(all_interests).most_common(10)
    print(f"Top 10 관심사: {interest_counts}")
    
    # 6. Negative Type Distribution
    print("\n--- 6. Negative 유형 분포 ---")
    neg_data = [d for d in data if d.get('pairType') == 'negative']
    rejection_types = Counter(d.get('rejectionType') for d in neg_data)
    print(f"거절 유형: {dict(rejection_types)}")
    
    # 7. Match Quality Score (Positive)
    print("\n--- 7. Positive 매칭 품질 점수 ---")
    pos_data = [d for d in data if d.get('pairType') == 'positive']
    if pos_data:
        scores = [d.get('matchQualityScore', 0) for d in pos_data if d.get('matchQualityScore') is not None]
        if scores:
            print(f"평균 품질 점수: {sum(scores)/len(scores):.3f}")
            print(f"최소: {min(scores):.3f}, 최대: {max(scores):.3f}")
    
    # 8. Local Image Paths Check
    print("\n--- 8. 로컬 이미지 경로 존재 여부 ---")
    has_local_paths = 0
    for d in data:
        ua = d.get('userA', {})
        ub = d.get('userB', {})
        if 'localImagePaths' in ua or 'localImagePaths' in ub:
            has_local_paths += 1
    print(f"localImagePaths 필드 포함 건수: {has_local_paths}")
    
    # 9. Potential Issues
    print("\n--- 9. 잠재적 이슈 점검 ---")
    issues = []
    
    # Check for missing required fields
    required_fields = ['pairId', 'pairType', 'userA', 'userB', 'label']
    for i, d in enumerate(data):
        for field in required_fields:
            if field not in d:
                issues.append(f"Row {i}: Missing '{field}'")
    
    # Check for empty images
    for i, d in enumerate(data):
        ua = d.get('userA', {})
        ub = d.get('userB', {})
        if not ua.get('imagePaths') or not ub.get('imagePaths'):
            issues.append(f"Row {i} ({d.get('pairId')}): Empty imagePaths")
    
    if issues:
        print(f"발견된 이슈: {len(issues)}건")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... 외 {len(issues) - 10}건")
    else:
        print("이슈 없음 ✓")
    
    print("\n=== 검토 완료 ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", type=str, default="/home/sagemaker-user/data/dataset.jsonl")
    args = parser.parse_args()
    analyze_dataset(args.dataset_jsonl)
