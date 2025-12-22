"""
데이터 분할 스크립트

데이터 디렉토리를 스캔하여 train/valid/test 분할 JSON 파일을 생성합니다.

사용법:
    python scripts/prepare_splits.py --data-dir /path/to/data
    python scripts/prepare_splits.py --train-ratio 0.7 --valid-ratio 0.15 --test-ratio 0.15
"""

import os
import json
import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="데이터 분할 JSON 생성")
    parser.add_argument("--data-dir", type=str, 
                        default=os.path.expanduser("~/data/mutual-like-validations/images"),
                        help="커플 이미지 디렉토리")
    parser.add_argument("--output", type=str, default="couple_splits.json",
                        help="출력 JSON 파일 경로")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="학습 비율")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="검증 비율")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="테스트 비율")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()
    
    # 비율 검증
    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"경고: 비율 합계가 1이 아닙니다 ({total_ratio})")
    
    # 커플 ID 수집
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"오류: 디렉토리가 존재하지 않습니다: {args.data_dir}")
        return 1
    
    couple_dirs = sorted([d.name for d in data_path.iterdir() 
                          if d.is_dir() and d.name.startswith("couple_")])
    all_ids = [int(d.split("_")[1]) for d in couple_dirs]
    
    print(f"총 {len(all_ids)}개의 커플 발견")
    
    # 셔플 및 분할
    random.seed(args.seed)
    random.shuffle(all_ids)
    
    n_total = len(all_ids)
    n_train = int(n_total * args.train_ratio)
    n_valid = int(n_total * args.valid_ratio)
    
    train_ids = all_ids[:n_train]
    valid_ids = all_ids[n_train:n_train + n_valid]
    test_ids = all_ids[n_train + n_valid:]
    
    # JSON 저장
    splits = {
        "train": sorted(train_ids),
        "valid": sorted(valid_ids),
        "test": sorted(test_ids),
        "config": {
            "train_ratio": args.train_ratio,
            "valid_ratio": args.valid_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "total_couples": n_total
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n분할 완료:")
    print(f"  - Train: {len(train_ids)}개 ({len(train_ids)/n_total*100:.1f}%)")
    print(f"  - Valid: {len(valid_ids)}개 ({len(valid_ids)/n_total*100:.1f}%)")
    print(f"  - Test:  {len(test_ids)}개 ({len(test_ids)/n_total*100:.1f}%)")
    print(f"\n저장됨: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
