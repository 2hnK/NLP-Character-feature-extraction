"""
로컬 테스트 이미지 Gemini 라벨링 스크립트

로컬 폴더의 이미지들을 Gemini API로 라벨링하여 JSONL 파일로 저장합니다.
학습에 사용된 5가지 fashion_style 분류를 사용합니다.

사용법:
    set GOOGLE_API_KEY=your_api_key
    python scripts/labeling/label_local_images.py
"""

import os
import json
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# .env 파일 로드
load_dotenv()

# 학습에 사용된 5가지 fashion_style (label_mapping.json과 일치)
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
    # 입력
    image_dir: str = "archive/test_images"  # 테스트 이미지 폴더
    
    # 출력
    output_jsonl: str = "archive/validation_labeled.jsonl"
    
    # Gemini 설정
    model_name: str = "gemini-2.5-flash"
    
    # 처리 설정
    delay_between_requests: float = 0.5  # API 요청 간 대기 시간 (초)


def process_image(model: genai.GenerativeModel, image_path: Path) -> dict:
    """단일 이미지 처리"""
    try:
        image = Image.open(image_path).convert('RGB')
        
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
        
        # fashion_style 검증 및 수정
        if metadata.get('fashion_style') not in FASHION_STYLES:
            print(f"  ⚠️ Invalid fashion_style: {metadata.get('fashion_style')}, defaulting to Casual_Basic")
            metadata['fashion_style'] = 'Casual_Basic'
        
        return metadata
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def main():
    config = Config()
    
    # API 키 확인
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Set it with: set GOOGLE_API_KEY=your_api_key")
        return
    
    # Gemini 초기화
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.model_name)
    print(f"✅ Gemini model loaded: {config.model_name}")
    
    # 이미지 목록 가져오기
    image_dir = Path(config.image_dir)
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return
    
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    print(f"📁 Found {len(image_files)} images in {image_dir}")
    
    if not image_files:
        print("No images found to process.")
        return
    
    # 기존 처리 항목 로드 (이어서 처리)
    processed_filenames = set()
    output_path = Path(config.output_jsonl)
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_filenames.add(data.get('filename'))
                except:
                    pass
        print(f"📋 Already processed: {len(processed_filenames)} items")
    
    # 처리 시작
    results = []
    errors = []
    
    with open(output_path, 'a', encoding='utf-8') as output_file:
        for image_path in tqdm(image_files, desc="Processing images"):
            filename = image_path.name
            
            # 이미 처리된 항목 건너뛰기
            if filename in processed_filenames:
                continue
            
            # 이미지 처리
            metadata = process_image(model, image_path)
            
            if metadata:
                result = {
                    'id': f"test_{image_path.stem}",
                    'filename': filename,
                    'image_metadata': metadata
                }
                
                # 즉시 파일에 저장
                output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                output_file.flush()
                results.append(result)
            else:
                errors.append(filename)
            
            # Rate limiting
            time.sleep(config.delay_between_requests)
    
    # 결과 요약
    print(f"\n{'='*60}")
    print(f"✅ Processing Complete!")
    print(f"{'='*60}")
    print(f"Total images: {len(image_files)}")
    print(f"Newly processed: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Output saved to: {output_path}")
    
    if errors:
        print(f"\nFailed images: {errors}")
    
    # 클래스 분포 출력
    if results:
        style_counts = {}
        for r in results:
            style = r['image_metadata'].get('fashion_style', 'Unknown')
            style_counts[style] = style_counts.get(style, 0) + 1
        
        print(f"\n📊 Class Distribution:")
        for style, count in sorted(style_counts.items()):
            print(f"  - {style}: {count}")


if __name__ == "__main__":
    main()
