import json
import os

"""
라벨 통일 및 정규화 스크립트

이 스크립트는 정의된 매핑 규칙에 따라 메타데이터 라벨을 수정합니다.
주요 작업:
1. Fashion Style, Vibe Category, Shot Type의 불일치하거나 중복된 라벨을 표준 라벨로 매핑.
2. 원본 라벨을 `_original_*` 필드로 보존하여 추적 가능하게 함.

사용법:
    python fix_labels.py
"""

def fix_labels(input_file, output_file):
    print(f"🔧 라벨 통일 작업 시작: {input_file} -> {output_file}")

    if not os.path.exists(input_file):
        print(f"❌ 오류: 입력 파일을 찾을 수 없습니다: {input_file}")
        return

    # 매핑 정의 (Final 8/5/7 Schema 기반)
    mapping = {
        "fashion_style": {
            "Business_Casual": "Dandy_Minimal",
            "Elegant_Chic": "Chic_Modern",
            "Sporty_Active": "Sporty_Athleisure",
            "Street_Casual": "Street_Hip",      # 복원된 라벨로 매핑
            "Traditional_Korean": "Street_Hip", # Vintage_Retro 데이터 부족으로 Street_Hip으로 통합
            "Vintage_Retro": "Street_Hip",      # 데이터 부족으로 Street_Hip으로 통합
            "Trendy_Fashion": "Chic_Modern",
            "Cool_Charismatic": "Street_Hip"    # 잘못된 라벨 수정
        },
        "vibe_category": {
            "Artistic_Creative": "Artistic_Unique",
            "Casual_Relaxed": "Warm_Friendly",
            "Cool_Mysterious": "Cool_Charismatic",
            "Energetic_Playful": "Energetic_Active",
            "Intellectual_Quiet": "Professional_Smart",
            "Professional_Confident": "Professional_Smart",
            "Romantic_Charming": "Warm_Friendly",
            "Sophisticated_Elegant": "Professional_Smart", # Elegant_Luxury가 학습에 없으므로 대체
            "Casual_Basic": "Warm_Friendly",
            "Chic_Modern": "Professional_Smart",
            "Feminine_Romantic": "Warm_Friendly",
            "Sporty_Active": "Energetic_Active"
        },
        "shot_type": {
            "Action_Shot": "FullBody_Shot",
            "Casual_Snapshot": "Portrait_OtherTaken",
            "Half_Body": "Portrait_OtherTaken",
            "Outdoor_Natural": "Portrait_OtherTaken",
            "Professional_Studio": "Portrait_OtherTaken",
            "Portrait_Selfie": "Mirrored_Selfie" # 복원된 라벨로 매핑
        }
    }

    fixed_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip():
                continue
            
            total_count += 1
            item = json.loads(line)
            meta = item.get('image_metadata', {})
            
            is_modified = False
            
            # 1. Fashion Style 매핑
            style = meta.get('fashion_style')
            if style in mapping["fashion_style"]:
                meta['fashion_style'] = mapping["fashion_style"][style]
                # 원본 라벨 보존 (디버깅용)
                meta['_original_fashion_style'] = style
                is_modified = True
                
            # 2. Vibe Category 매핑
            vibe = meta.get('vibe_category')
            if vibe in mapping["vibe_category"]:
                meta['vibe_category'] = mapping["vibe_category"][vibe]
                meta['_original_vibe_category'] = vibe
                is_modified = True
                
            # 3. Shot Type 매핑
            shot = meta.get('shot_type')
            if shot in mapping["shot_type"]:
                meta['shot_type'] = mapping["shot_type"][shot]
                meta['_original_shot_type'] = shot
                is_modified = True
            
            if is_modified:
                fixed_count += 1
                item['image_metadata'] = meta
            
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ 작업 완료!")
    print(f"   - 총 처리된 항목: {total_count}")
    print(f"   - 수정된 항목: {fixed_count}")
    print(f"   - 저장된 파일: {output_file}")

if __name__ == "__main__":
    fix_labels('train_valid.jsonl', 'train_valid_fixed.jsonl')
