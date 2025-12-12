"""
커플 데이터 분할 스크립트

775쌍의 커플 데이터를 Train+Valid(80%)와 Test(20%)로 분할하고,
Train+Valid 내에서 5-Fold Cross Validation 인덱스를 생성합니다.

출력 파일:
- couple_splits.json: 전체 분할 정보
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sklearn.model_selection import KFold


def find_valid_couples(data_dir: str, start: int = 5, end: int = 778) -> List[int]:
    """유효한 커플 폴더 찾기"""
    data_path = Path(data_dir)
    valid_couples = []
    
    for couple_num in range(start, end + 1):
        couple_dir = data_path / f"couple_{couple_num}"
        female_path = couple_dir / "female.png"
        male_path = couple_dir / "male.png"
        
        if female_path.exists() and male_path.exists():
            valid_couples.append(couple_num)
    
    return valid_couples


def create_splits(
    couple_ids: List[int],
    test_ratio: float = 0.2,
    n_folds: int = 5,
    seed: int = 42
) -> Dict:
    """데이터 분할 생성"""
    random.seed(seed)
    np.random.seed(seed)
    
    # 셔플
    shuffled_ids = couple_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Test set 분리
    n_test = int(len(shuffled_ids) * test_ratio)
    test_ids = shuffled_ids[:n_test]
    train_valid_ids = shuffled_ids[n_test:]
    
    # 5-Fold CV 인덱스 생성
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    
    train_valid_array = np.array(train_valid_ids)
    for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(train_valid_array)):
        folds.append({
            "fold": fold_idx,
            "train": train_valid_array[train_idx].tolist(),
            "valid": train_valid_array[valid_idx].tolist()
        })
    
    return {
        "seed": seed,
        "total_couples": len(couple_ids),
        "test_couples": len(test_ids),
        "train_valid_couples": len(train_valid_ids),
        "n_folds": n_folds,
        "test": test_ids,
        "folds": folds
    }


def main():
    parser = argparse.ArgumentParser(description="Split couple data for training")
    parser.add_argument("--data-dir", type=str, 
                        default=os.path.expanduser("~/data/mutual-like-validations/images"),
                        help="Couple data directory")
    parser.add_argument("--output", type=str, default="couple_splits.json",
                        help="Output JSON file")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Test set ratio")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 커플 데이터 분할")
    print("=" * 60)
    
    # 유효한 커플 찾기
    print(f"\n데이터 디렉토리: {args.data_dir}")
    valid_couples = find_valid_couples(args.data_dir)
    print(f"유효한 커플 수: {len(valid_couples)}")
    
    if not valid_couples:
        print("❌ 유효한 커플 데이터가 없습니다.")
        return 1
    
    # 분할 생성
    splits = create_splits(
        valid_couples,
        test_ratio=args.test_ratio,
        n_folds=args.n_folds,
        seed=args.seed
    )
    
    # 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    
    # 결과 출력
    print(f"\n📋 분할 결과:")
    print(f"  - 총 커플: {splits['total_couples']}")
    print(f"  - Test Set: {splits['test_couples']} ({args.test_ratio*100:.0f}%)")
    print(f"  - Train+Valid: {splits['train_valid_couples']} ({(1-args.test_ratio)*100:.0f}%)")
    print(f"  - Folds: {splits['n_folds']}")
    
    for fold in splits['folds']:
        print(f"    Fold {fold['fold']}: Train {len(fold['train'])}, Valid {len(fold['valid'])}")
    
    print(f"\n✅ 저장 완료: {args.output}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
