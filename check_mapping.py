import json
from collections import Counter, defaultdict
import os

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
        
        # Mapping 파일 저장 (Main Label인 fashion_style만 저장하거나, 필요시 모두 저장)
        if field == 'fashion_style':
            mapping = {name: i for i, name in enumerate(all_classes)}
            with open('label_mapping.json', 'w') as f:
                json.dump(mapping, f, indent=4)
            print(f"💾 [fashion_style] 매핑 파일 저장됨: label_mapping.json")

if __name__ == "__main__":
    check_all_statistics('train_aug.jsonl', 'train_valid.jsonl')