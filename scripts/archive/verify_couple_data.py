"""
S3 커플 데이터 무결성 검증 스크립트

couple_1 ~ couple_778 폴더가 모두 존재하는지,
각 폴더에 female.png, male.png가 있는지 검증합니다.
"""

import boto3
import logging
from typing import List, Tuple, Dict
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# S3 설정
BUCKET_NAME = "sagemaker-ap-northeast-2-369036988146"
PREFIX = "data/mutual-like-validations/images/"
TOTAL_COUPLES = 778
REQUIRED_FILES = ["female.png", "male.png"]


def verify_couple_data() -> Dict:
    """S3 커플 데이터 무결성 검증"""
    s3_client = boto3.client('s3')
    
    results = {
        "total_expected": TOTAL_COUPLES,
        "valid_couples": [],
        "missing_folders": [],
        "incomplete_couples": [],  # 폴더는 있지만 파일 누락
    }
    
    logger.info(f"Verifying {TOTAL_COUPLES} couples in s3://{BUCKET_NAME}/{PREFIX}")
    
    for couple_num in tqdm(range(1, TOTAL_COUPLES + 1), desc="Verifying couples"):
        folder_prefix = f"{PREFIX}couple_{couple_num}/"
        
        try:
            # 폴더 내 파일 목록 조회
            response = s3_client.list_objects_v2(
                Bucket=BUCKET_NAME,
                Prefix=folder_prefix
            )
            
            if 'Contents' not in response:
                results["missing_folders"].append(couple_num)
                continue
            
            # 파일명 추출
            files_in_folder = [
                obj['Key'].split('/')[-1] 
                for obj in response['Contents']
                if obj['Key'] != folder_prefix  # 폴더 자체 제외
            ]
            
            # 필수 파일 확인
            missing_files = [f for f in REQUIRED_FILES if f not in files_in_folder]
            
            if missing_files:
                results["incomplete_couples"].append({
                    "couple_num": couple_num,
                    "missing_files": missing_files,
                    "found_files": files_in_folder
                })
            else:
                results["valid_couples"].append(couple_num)
                
        except Exception as e:
            logger.error(f"Error checking couple_{couple_num}: {e}")
            results["missing_folders"].append(couple_num)
    
    return results


def print_report(results: Dict):
    """검증 결과 리포트 출력"""
    print("\n" + "=" * 60)
    print("📊 S3 커플 데이터 검증 결과")
    print("=" * 60)
    
    valid_count = len(results["valid_couples"])
    missing_count = len(results["missing_folders"])
    incomplete_count = len(results["incomplete_couples"])
    total = results["total_expected"]
    
    print(f"\n✅ 유효한 커플: {valid_count}/{total} ({valid_count/total*100:.1f}%)")
    print(f"❌ 누락된 폴더: {missing_count}")
    print(f"⚠️ 불완전한 커플 (파일 누락): {incomplete_count}")
    
    if results["missing_folders"]:
        print(f"\n📁 누락된 폴더 목록:")
        # 연속된 범위로 표시
        ranges = []
        start = None
        prev = None
        for num in sorted(results["missing_folders"]):
            if start is None:
                start = prev = num
            elif num == prev + 1:
                prev = num
            else:
                ranges.append((start, prev))
                start = prev = num
        if start is not None:
            ranges.append((start, prev))
        
        for s, e in ranges[:10]:  # 처음 10개 범위만 표시
            if s == e:
                print(f"   - couple_{s}")
            else:
                print(f"   - couple_{s} ~ couple_{e}")
        if len(ranges) > 10:
            print(f"   ... 외 {len(ranges) - 10}개 범위")
    
    if results["incomplete_couples"]:
        print(f"\n⚠️ 불완전한 커플 상세:")
        for item in results["incomplete_couples"][:5]:  # 처음 5개만 표시
            print(f"   - couple_{item['couple_num']}: 누락={item['missing_files']}, 존재={item['found_files']}")
        if len(results["incomplete_couples"]) > 5:
            print(f"   ... 외 {len(results['incomplete_couples']) - 5}건")
    
    print("\n" + "=" * 60)
    
    return valid_count == total


def main():
    logger.info("Starting S3 couple data verification...")
    
    results = verify_couple_data()
    is_complete = print_report(results)
    
    # 결과 저장
    import json
    output_file = "couple_data_verification.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {output_file}")
    
    if is_complete:
        print("\n🎉 모든 커플 데이터가 정상입니다!")
        return 0
    else:
        print("\n⚠️ 일부 데이터가 누락되었습니다. 위 리포트를 확인하세요.")
        return 1


if __name__ == "__main__":
    exit(main())
