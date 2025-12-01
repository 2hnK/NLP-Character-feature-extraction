import json
from collections import Counter, defaultdict
import os

"""
메타데이터 통계 분석 및 라벨 매핑 생성 스크립트

이 스크립트는 Train/Valid 데이터셋의 메타데이터 분포를 분석하고 무결성을 검사합니다.
주요 기능:
1. 각 필드(Fashion Style, Vibe, Shot Type)별 클래스 분포 출력.
2. Train 데이터 부족(Low Data) 및 Valid Only(Unseen in Train) 클래스 감지.
3. `fashion_style`에 대한 라벨 매핑 파일(`label_mapping.json`) 생성.

사용법:
    python check_mapping.py [train_file] [valid_file]
"""

def check_all_statistics(train_file, valid_file):
    print(f"🔍 전체 메타데이터(Style, Vibe, Shot) 통계 분석 시작...\n")

    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        print(f"❌ 오류: 파일을 찾을 수 없습니다.")
        return

    # 분석할 타겟 필드 정의
    target_fields = ['fashion_style', 'vibe_category', 'shot_type']
    
    # 데이터 담을 구조: stats[field_name][train_or_valid] = [list of values]
    stats = {field: {'train': [], 'valid': []} for field in target_fields}

    def load_data(filename, split_name):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            meta = data.get('image_metadata', {})
                            
                            for field in target_fields:
                                value = meta.get(field)
                                if value:
                                    stats[field][split_name].append(value)
                        except: continue
        except Exception as e:
            print(f"⚠️ 파일 읽기 오류 ({filename}): {e}")

    # 데이터 로드
    load_data(train_file, 'train')
    load_data(valid_file, 'valid')

    # === 분석 및 출력 루프 ===
    for field in target_fields:
        print(f"\n{'='*20} [ {field.upper()} ] 분석 결과 {'='*20}")
        
        t_counter = Counter(stats[field]['train'])
        v_counter = Counter(stats[field]['valid'])
        
        all_classes = sorted(list(set(t_counter.keys()) | set(v_counter.keys())))
        
        print(f"{'Class Name':<30} | {'Train':<8} | {'Valid':<8} | {'Status'}")
        print("-" * 70)

        min_k = 4
        
        for cls in all_classes:
            t_cnt = t_counter.get(cls, 0)
            v_cnt = v_counter.get(cls, 0)
            
            status = ""
            
            # 1. Train 데이터 부족 체크
            if t_cnt < min_k:
                if field == 'fashion_style':
                    status = "🚨 CRITICAL (Sampler Error)"
                else:
                    status = "⚠️ Warning (Low Data)"
            
            # 2. Valid Only 체크 (학습 안 됨)
            if t_cnt == 0 and v_cnt > 0:
                status = "❌ Unseen in Train"

            print(f"{cls:<30} | {t_cnt:<8} | {v_cnt:<8} | {status}")

        print("-" * 70)
        print(f"Total Count: {len(stats[field]['train'])} (Train) / {len(stats[field]['valid'])} (Valid)")
        
import json
from collections import Counter, defaultdict
import os
import sys

def check_all_statistics(train_file, valid_file):
    output_lines = []
    output_lines.append(f"🔍 전체 메타데이터(Style, Vibe, Shot) 통계 분석 시작...\n")
    output_lines.append(f"📂 Train File: {train_file}")
    output_lines.append(f"📂 Valid File: {valid_file}")

    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        print(f"❌ 오류: 파일을 찾을 수 없습니다.")
        return

    # 분석할 타겟 필드 정의
    target_fields = ['fashion_style', 'vibe_category', 'shot_type']
    
    # 데이터 담을 구조: stats[field_name][train_or_valid] = [list of values]
    stats = {field: {'train': [], 'valid': []} for field in target_fields}

    def load_data(filename, split_name):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            meta = data.get('image_metadata', {})
                            
                            for field in target_fields:
                                value = meta.get(field)
                                if value:
                                    stats[field][split_name].append(value)
                        except: continue
        except Exception as e:
            output_lines.append(f"⚠️ 파일 읽기 오류 ({filename}): {e}")

    # 데이터 로드
    load_data(train_file, 'train')
    load_data(valid_file, 'valid')

    # === 분석 및 출력 루프 ===
    for field in target_fields:
        output_lines.append(f"\n{'='*20} [ {field.upper()} ] 분석 결과 {'='*20}")
        
        t_counter = Counter(stats[field]['train'])
        v_counter = Counter(stats[field]['valid'])
        
        all_classes = sorted(list(set(t_counter.keys()) | set(v_counter.keys())))
        
        output_lines.append(f"{'Class Name':<30} | {'Train':<8} | {'Valid':<8} | {'Status'}")
        output_lines.append("-" * 70)

        min_k = 4
        
        for cls in all_classes:
            t_cnt = t_counter.get(cls, 0)
            v_cnt = v_counter.get(cls, 0)
            
            status = ""
            
            # 1. Train 데이터 부족 체크
            if t_cnt < min_k:
                if field == 'fashion_style':
                    status = "🚨 CRITICAL (Sampler Error)"
                else:
                    status = "⚠️ Warning (Low Data)"
            
            # 2. Valid Only 체크 (학습 안 됨)
            if t_cnt == 0 and v_cnt > 0:
                status = "❌ Unseen in Train"

            output_lines.append(f"{cls:<30} | {t_cnt:<8} | {v_cnt:<8} | {status}")

        output_lines.append("-" * 70)
        output_lines.append(f"Total Count: {len(stats[field]['train'])} (Train) / {len(stats[field]['valid'])} (Valid)")
        
        # Mapping 파일 저장
        if field == 'fashion_style':
            mapping = {name: i for i, name in enumerate(all_classes)}
            with open('label_mapping.json', 'w') as f:
                json.dump(mapping, f, indent=4)
            output_lines.append(f"💾 [fashion_style] 매핑 파일 저장됨: label_mapping.json")

    # Print and Save
import json
from collections import Counter, defaultdict
import os
import sys

def check_all_statistics(train_file, valid_file):
    output_lines = []
    output_lines.append(f"🔍 전체 메타데이터(Style, Vibe, Shot) 통계 분석 시작...\n")
    output_lines.append(f"📂 Train File: {train_file}")
    output_lines.append(f"📂 Valid File: {valid_file}")

    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        print(f"❌ 오류: 파일을 찾을 수 없습니다.")
        return

    # 분석할 타겟 필드 정의
    target_fields = ['fashion_style', 'vibe_category', 'shot_type']
    
    # 데이터 담을 구조: stats[field_name][train_or_valid] = [list of values]
    stats = {field: {'train': [], 'valid': []} for field in target_fields}

    def load_data(filename, split_name):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            meta = data.get('image_metadata', {})
                            
                            for field in target_fields:
                                value = meta.get(field)
                                if value:
                                    stats[field][split_name].append(value)
                        except: continue
        except Exception as e:
            output_lines.append(f"⚠️ 파일 읽기 오류 ({filename}): {e}")

    # 데이터 로드
    load_data(train_file, 'train')
    load_data(valid_file, 'valid')

    # === 분석 및 출력 루프 ===
    for field in target_fields:
        output_lines.append(f"\n{'='*20} [ {field.upper()} ] 분석 결과 {'='*20}")
        
        t_counter = Counter(stats[field]['train'])
        v_counter = Counter(stats[field]['valid'])
        
        all_classes = sorted(list(set(t_counter.keys()) | set(v_counter.keys())))
        
        output_lines.append(f"{'Class Name':<30} | {'Train':<8} | {'Valid':<8} | {'Status'}")
        output_lines.append("-" * 70)

        min_k = 4
        
        for cls in all_classes:
            t_cnt = t_counter.get(cls, 0)
            v_cnt = v_counter.get(cls, 0)
            
            status = ""
            
            # 1. Train 데이터 부족 체크
            if t_cnt < min_k:
                if field == 'fashion_style':
                    status = "🚨 CRITICAL (Sampler Error)"
                else:
                    status = "⚠️ Warning (Low Data)"
            
            # 2. Valid Only 체크 (학습 안 됨)
            if t_cnt == 0 and v_cnt > 0:
                status = "❌ Unseen in Train"

            output_lines.append(f"{cls:<30} | {t_cnt:<8} | {v_cnt:<8} | {status}")

        output_lines.append("-" * 70)
        output_lines.append(f"Total Count: {len(stats[field]['train'])} (Train) / {len(stats[field]['valid'])} (Valid)")
        
        # Mapping 파일 저장
        if field == 'fashion_style':
            mapping = {name: i for i, name in enumerate(all_classes)}
            with open('label_mapping.json', 'w') as f:
                json.dump(mapping, f, indent=4)
            output_lines.append(f"💾 [fashion_style] 매핑 파일 저장됨: label_mapping.json")

    # Print and Save
    full_report = '\n'.join(output_lines)
    print(full_report)
    with open('report.txt', 'w', encoding='utf-8') as f:
        f.write(full_report)

if __name__ == "__main__":
    train_file = sys.argv[1] if len(sys.argv) > 1 else 'train_aug_final.jsonl'
    valid_file = sys.argv[2] if len(sys.argv) > 2 else 'train_valid_final.jsonl'
    check_all_statistics(train_file, valid_file)