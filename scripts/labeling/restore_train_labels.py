import json
import os
import shutil

"""
학습 데이터 라벨 복원 스크립트

이 스크립트는 잘못 매핑되거나 손실된 특정 라벨을 원본 메타데이터(`_original_*`)를 기반으로 복원합니다.
주요 복원 대상:
1. `Street_Hip` 스타일 (Cool_Charismatic 오분류 수정).
2. `Mirrored_Selfie` 샷 타입.
3. `FullBody_Shot` 표기 정규화.

사용법:
    python restore_train_labels.py
"""

def restore_train_labels(input_file, output_file):
    print(f"🔧 학습 데이터 라벨 복원 시작: {input_file} -> {output_file}")

    if not os.path.exists(input_file):
        print(f"❌ 오류: 입력 파일을 찾을 수 없습니다: {input_file}")
        return

    restored_count = 0
    fixed_cool_count = 0
    total_count = 0
    
    # 백업 생성
    backup_file = input_file + ".bak"
    if not os.path.exists(backup_file):
        shutil.copy(input_file, backup_file)
        print(f"📦 백업 파일 생성됨: {backup_file}")

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip():
                continue
            
            total_count += 1
            item = json.loads(line)
            meta = item.get('image_metadata', {})
            
            is_modified = False
            
            # 1. Street_Hip 복원 (Cool_Charismatic 오분류 수정 포함)
            # 조건: 현재 fashion_style이 Cool_Charismatic이거나, _original_fashion_style이 Street_Hip인 경우
            current_fashion = meta.get('fashion_style')
            original_fashion = meta.get('_original_fashion_style')
            
            if original_fashion == 'Street_Hip':
                if current_fashion != 'Street_Hip':
                    meta['fashion_style'] = 'Street_Hip'
                    is_modified = True
                    restored_count += 1
            elif current_fashion == 'Cool_Charismatic': # 원본이 Street_Hip이 아니더라도 Cool_Charismatic이 Fashion에 있으면 수정 필요
                 # Cool_Charismatic은 Vibe에만 있어야 함. Fashion에서는 Street_Hip으로 매핑 (사용자 합의)
                 meta['fashion_style'] = 'Street_Hip'
                 is_modified = True
                 fixed_cool_count += 1

            # 2. Mirrored_Selfie 복원
            current_shot = meta.get('shot_type')
            original_shot = meta.get('_original_shot_type')
            
            if original_shot == 'Mirrored_Selfie':
                if current_shot != 'Mirrored_Selfie':
                    meta['shot_type'] = 'Mirrored_Selfie'
                    is_modified = True
                    restored_count += 1
            
            # 3. FullBody_Shot 정규화 (Full_Body_Shot -> FullBody_Shot)
            if meta.get('shot_type') == 'Full_Body_Shot':
                meta['shot_type'] = 'FullBody_Shot'
                is_modified = True
                restored_count += 1
            
            if is_modified:
                item['image_metadata'] = meta
            
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ 작업 완료!")
    print(f"   - 총 처리된 항목: {total_count}")
    print(f"   - 복원/수정된 항목(중복 포함): {restored_count + fixed_cool_count}")
    print(f"   - 저장된 파일: {output_file}")

if __name__ == "__main__":
    # 안전을 위해 새 파일에 쓰고, 확인 후 덮어쓰는 방식을 권장하지만, 
    # 여기서는 바로 train_aug_restored.jsonl로 저장.
    restore_train_labels('train_aug.jsonl', 'train_aug_restored.jsonl')
