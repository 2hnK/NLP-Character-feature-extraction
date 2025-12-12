"""
커플 이미지 Gemini 라벨링 스크립트

기존 학습 데이터와 동일한 방식(Gemini-2.5-flash)으로
커플 이미지 775쌍에 스타일 라벨을 부여합니다.

출력: couple_labels.jsonl
"""

import os
import sys
import json
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# 학습에 사용된 5가지 fashion_style
FASHION_STYLES = [
    "Casual_Basic",
    "Street_Hip", 
    "Sporty_Athleisure",
    "Chic_Modern",
    "Classy_Elegant"
]

SYSTEM_PROMPT = f"""
Role: Expert Fashion & Social Profile Analyst for Dating Apps.
Output Language: ENGLISH ONLY.

Task: Extract structured metadata from the profile image.

--- 1. CATEGORICAL FIELDS (Strict Enums) ---
Choose exactly ONE value for each category based on visual evidence.

A. fashion_style (MUST be exactly one of these):
- "Casual_Basic": Comfortable, t-shirts, jeans, hoodies, relaxed fit.
- "Street_Hip": Oversized, layering, trendy, cargo, hip-hop vibe.
- "Sporty_Athleisure": Gym wear, leggings, jerseys, tracksuits, activewear.
- "Chic_Modern": All-black, leather, sharp, edgy, city vibe, cool.
- "Classy_Elegant": Shirts, slacks, suits, blouses, dresses, coats, neat & formal.

B. shot_type:
- "Selfie_CloseUp": Face-focused selfie (holding camera).
- "Mirrored_Selfie": Full body or half body selfie taken in a mirror.
- "Others_Cam": Shot taken by someone else (candid, portrait, full body).

C. visual_quality:
- "High" (Pro/Studio/Clear), "Medium" (Decent Mobile), "Low" (Blurry/Dark)

--- 2. DESCRIPTIVE FIELDS (Text) ---

D. physical_features (List of Strings):
- List 3-5 distinct visual traits focusing on hair, accessories, and grooming. 
- Examples: "Wavy brown hair", "Rimless glasses", "Drop earrings", "Beanie hat", "Red lipstick".
- Do NOT describe clothing here (already covered in style).

E. caption (String):
- A natural language description (1-2 sentences) summarizing the person's appearance and the scene.
- Used for text-to-image training.

--- Output JSON Format ---
{{
  "fashion_style": "Enum Value (one of: {', '.join(FASHION_STYLES)})",
  "shot_type": "Enum Value",
  "visual_quality": "Enum Value",
  "physical_features": ["trait1", "trait2", ...],
  "caption": "Full sentence description."
}}
"""


@dataclass
class Config:
    """설정"""
    data_dir: str = os.path.expanduser("~/data/mutual-like-validations/images")
    output_jsonl: str = "couple_labels.jsonl"
    model_name: str = "gemini-2.5-flash"
    delay_between_requests: float = 0.3
    start_couple: int = 5
    end_couple: int = 778


def process_image(model: genai.GenerativeModel, image_path: Path) -> dict:
    """단일 이미지 처리"""
    try:
        image = Image.open(image_path).convert('RGB')
        
        # 이미지 크기 제한 (API 요청 최적화)
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.BICUBIC)
        
        response = model.generate_content(
            [SYSTEM_PROMPT, image],
            generation_config={"response_mime_type": "application/json"},
            request_options={"timeout": 120},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        metadata = json.loads(response.text)
        
        # fashion_style 검증
        if metadata.get('fashion_style') not in FASHION_STYLES:
            print(f"  ⚠️ Invalid fashion_style: {metadata.get('fashion_style')}")
            metadata['fashion_style'] = 'Casual_Basic'
        
        return metadata
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="couple_labels.jsonl")
    parser.add_argument("--start", type=int, default=5)
    parser.add_argument("--end", type=int, default=778)
    args = parser.parse_args()
    
    config = Config()
    if args.data_dir:
        config.data_dir = args.data_dir
    config.output_jsonl = args.output
    config.start_couple = args.start
    config.end_couple = args.end
    
    # API 키 확인
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set")
        print("Set it with: export GOOGLE_API_KEY=your_api_key")
        return 1
    
    # Gemini 초기화
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.model_name)
    print(f"✅ Gemini model: {config.model_name}")
    
    data_dir = Path(config.data_dir)
    output_path = Path(config.output_jsonl)
    
    # 기존 처리 항목 로드
    processed = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed.add(data.get('couple_id'))
                except:
                    pass
        print(f"📋 Already processed: {len(processed)} couples")
    
    # 라벨링 시작
    errors = []
    new_count = 0
    
    with open(output_path, 'a', encoding='utf-8') as out_file:
        for couple_num in tqdm(range(config.start_couple, config.end_couple + 1), desc="Labeling"):
            if couple_num in processed:
                continue
            
            couple_dir = data_dir / f"couple_{couple_num}"
            female_path = couple_dir / "female.png"
            male_path = couple_dir / "male.png"
            
            if not female_path.exists() or not male_path.exists():
                errors.append(couple_num)
                continue
            
            # Female 라벨링
            female_meta = process_image(model, female_path)
            time.sleep(config.delay_between_requests)
            
            # Male 라벨링
            male_meta = process_image(model, male_path)
            time.sleep(config.delay_between_requests)
            
            if female_meta and male_meta:
                result = {
                    'couple_id': couple_num,
                    'female': female_meta,
                    'male': male_meta,
                    'style_match': female_meta.get('fashion_style') == male_meta.get('fashion_style')
                }
                
                out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                out_file.flush()
                new_count += 1
            else:
                errors.append(couple_num)
    
    # 결과 요약
    print(f"\n{'='*60}")
    print(f"✅ Labeling Complete!")
    print(f"{'='*60}")
    print(f"Newly labeled: {new_count}")
    print(f"Errors: {len(errors)}")
    print(f"Output: {output_path}")
    
    # 스타일 일치율 계산
    if output_path.exists():
        match_count = 0
        total_count = 0
        style_dist = {}
        
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                total_count += 1
                if data.get('style_match'):
                    match_count += 1
                
                f_style = data['female'].get('fashion_style', 'Unknown')
                m_style = data['male'].get('fashion_style', 'Unknown')
                style_dist[f_style] = style_dist.get(f_style, 0) + 1
                style_dist[m_style] = style_dist.get(m_style, 0) + 1
        
        print(f"\n📊 스타일 일치율: {match_count}/{total_count} ({match_count/total_count*100:.1f}%)")
        print(f"   랜덤 기대값: 20%")
        
        print(f"\n📊 스타일 분포:")
        for style, count in sorted(style_dist.items(), key=lambda x: -x[1]):
            print(f"   - {style}: {count}")
    
    return 0


if __name__ == "__main__":
    exit(main())
