import json
import os
from collections import OrderedDict

"""
S3 경로 정렬 및 JSONL 스키마 통일 스크립트

이 스크립트는 다음 작업을 수행합니다:
1. Train 데이터: `image_filename` (aug_XXXXX.jpg) 추가 및 S3 경로(`image_path`) 생성.
2. Valid 데이터: `image_path`를 S3 구조에 맞게 업데이트.
3. 스키마 통일: 모든 JSONL 항목의 키 순서를 `id`, `image_path`, `image_filename`, `image_metadata` 순으로 정렬.

사용법:
    python align_paths.py
"""

def reorder_item(item, filename, s3_path_prefix):
    """
    Enforce key order: id, image_path, image_filename, image_metadata, ...
    """
    new_item = OrderedDict()
    
    # 1. ID
    if 'id' in item:
        new_item['id'] = item['id']
        
    # 2. Image Path & Filename
    # Construct full S3 path
    full_path = f"{s3_path_prefix}/{filename}"
    new_item['image_path'] = full_path
    new_item['image_filename'] = filename
    
    # 3. Image Metadata
    if 'image_metadata' in item:
        new_item['image_metadata'] = item['image_metadata']
        
    # 4. Others (preserve remaining keys)
    for k, v in item.items():
        if k not in ['id', 'image_path', 'image_filename', 'image_metadata']:
            new_item[k] = v
            
    return new_item

def align_paths(train_input, train_output, valid_input, valid_output):
    print(f"🔧 경로 정렬 및 스키마 통일 작업 시작...")

    # 1. Train Data 처리
    # Prefix: sometimes-ki-datasets/dataset/qwen-vl-train-v1/images
    train_prefix = "sometimes-ki-datasets/dataset/qwen-vl-train-v1/images"
    print(f"📂 Processing Train: {train_input} -> {train_output}")
    
    with open(train_input, 'r', encoding='utf-8') as fin, \
         open(train_output, 'w', encoding='utf-8') as fout:
        
        for i, line in enumerate(fin):
            if not line.strip():
                continue
            
            item = json.loads(line)
            
            # 파일명 생성 (aug_00000.jpg 형식)
            filename = f"aug_{i:05d}.jpg"
            
            # Reorder and inject paths
            new_item = reorder_item(item, filename, train_prefix)
            
            fout.write(json.dumps(new_item, ensure_ascii=False) + '\n')
            
    print(f"✅ Train 완료: {i+1}개 항목 처리됨.")

    # 2. Valid Data 처리
    # Prefix: sometimes-ki-datasets/dataset/validation/images
    valid_prefix = "sometimes-ki-datasets/dataset/validation/images"
    print(f"📂 Processing Valid: {valid_input} -> {valid_output}")
    
    with open(valid_input, 'r', encoding='utf-8') as fin, \
         open(valid_output, 'w', encoding='utf-8') as fout:
        
        for i, line in enumerate(fin):
            if not line.strip():
                continue
            
            item = json.loads(line)
            
            # 기존 filename 확인 또는 생성
            filename = item.get('image_filename')
            if not filename:
                filename = f"val_{i:05d}.jpg"
            
            # Reorder and inject paths
            new_item = reorder_item(item, filename, valid_prefix)
            
            fout.write(json.dumps(new_item, ensure_ascii=False) + '\n')

    print(f"✅ Valid 완료: {i+1}개 항목 처리됨.")
    print(f"🎉 모든 작업 완료! (Schema Unified)")

if __name__ == "__main__":
    align_paths(
        'train_aug_restored.jsonl', 'train_aug_final.jsonl',
        'train_valid_fixed.jsonl', 'train_valid_final.jsonl'
    )
