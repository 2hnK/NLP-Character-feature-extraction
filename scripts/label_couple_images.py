"""
커플 이미지 비동기 Gemini 라벨링 스크립트

asyncio + aiohttp를 사용하여 병렬 라벨링 처리
기존 대비 5-10배 빠른 처리 속도
"""

import os
import sys
import json
import asyncio
import aiofiles
from pathlib import Path
from PIL import Image
from tqdm.asyncio import tqdm_asyncio
from dataclasses import dataclass
import io
import base64

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

FASHION_STYLES = [
    "Casual_Basic", "Street_Hip", "Sporty_Athleisure",
    "Chic_Modern", "Classy_Elegant"
]

SYSTEM_PROMPT = f"""
Role: Expert Fashion & Social Profile Analyst for Dating Apps.
Output Language: ENGLISH ONLY.

Task: Extract structured metadata from the profile image.

--- 1. CATEGORICAL FIELDS (Strict Enums) ---
A. fashion_style (MUST be exactly one of these):
- "Casual_Basic": Comfortable, t-shirts, jeans, hoodies, relaxed fit.
- "Street_Hip": Oversized, layering, trendy, cargo, hip-hop vibe.
- "Sporty_Athleisure": Gym wear, leggings, jerseys, tracksuits, activewear.
- "Chic_Modern": All-black, leather, sharp, edgy, city vibe, cool.
- "Classy_Elegant": Shirts, slacks, suits, blouses, dresses, coats, neat & formal.

B. shot_type:
- "Selfie_CloseUp", "Mirrored_Selfie", "Others_Cam"

C. visual_quality:
- "High", "Medium", "Low"

--- 2. DESCRIPTIVE FIELDS ---
D. physical_features: List 3-5 traits (hair, accessories).
E. caption: 1-2 sentence description.

--- Output JSON ---
{{"fashion_style": "Enum", "shot_type": "Enum", "visual_quality": "Enum", "physical_features": [...], "caption": "..."}}
"""


@dataclass
class Config:
    data_dir: str = os.path.expanduser("~/data/mutual-like-validations/images")
    output_jsonl: str = "couple_labels.jsonl"
    model_name: str = "gemini-2.5-flash"
    max_concurrent: int = 10  # 동시 요청 수
    start_couple: int = 5
    end_couple: int = 778


def resize_image(image_path: Path, max_size: int = 512) -> Image.Image:
    """이미지 리사이즈 (API 요청 최적화)"""
    img = Image.open(image_path).convert('RGB')
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.BICUBIC)
    return img


async def process_single_image(model, image_path: Path, semaphore) -> dict:
    """단일 이미지 비동기 처리"""
    async with semaphore:
        try:
            img = resize_image(image_path)
            
            response = await asyncio.to_thread(
                model.generate_content,
                [SYSTEM_PROMPT, img],
                generation_config={"response_mime_type": "application/json"},
                request_options={"timeout": 60},
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            metadata = json.loads(response.text)
            
            if metadata.get('fashion_style') not in FASHION_STYLES:
                metadata['fashion_style'] = 'Casual_Basic'
            
            return metadata
            
        except Exception as e:
            print(f"Error {image_path.name}: {e}")
            return None


async def process_couple(model, couple_id: int, data_dir: Path, semaphore) -> dict:
    """커플 쌍 비동기 처리"""
    couple_dir = data_dir / f"couple_{couple_id}"
    female_path = couple_dir / "female.png"
    male_path = couple_dir / "male.png"
    
    if not female_path.exists() or not male_path.exists():
        return None
    
    # 두 이미지 동시 처리
    female_meta, male_meta = await asyncio.gather(
        process_single_image(model, female_path, semaphore),
        process_single_image(model, male_path, semaphore)
    )
    
    if female_meta and male_meta:
        return {
            'couple_id': couple_id,
            'female': female_meta,
            'male': male_meta,
            'style_match': female_meta.get('fashion_style') == male_meta.get('fashion_style')
        }
    return None


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="couple_labels.jsonl")
    parser.add_argument("--concurrent", type=int, default=10)
    parser.add_argument("--start", type=int, default=5)
    parser.add_argument("--end", type=int, default=778)
    args = parser.parse_args()
    
    config = Config()
    if args.data_dir:
        config.data_dir = args.data_dir
    config.output_jsonl = args.output
    config.max_concurrent = args.concurrent
    config.start_couple = args.start
    config.end_couple = args.end
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set")
        return 1
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.model_name)
    print(f"✅ Gemini model: {config.model_name}")
    print(f"🚀 Concurrent requests: {config.max_concurrent}")
    
    data_dir = Path(config.data_dir)
    output_path = Path(config.output_jsonl)
    
    # 기존 처리 항목 로드
    processed = set()
    if output_path.exists():
        async with aiofiles.open(output_path, 'r', encoding='utf-8') as f:
            async for line in f:
                try:
                    data = json.loads(line)
                    processed.add(data.get('couple_id'))
                except:
                    pass
        print(f"📋 Already processed: {len(processed)} couples")
    
    # 처리할 커플 ID 목록
    couple_ids = [i for i in range(config.start_couple, config.end_couple + 1) 
                  if i not in processed]
    print(f"📦 To process: {len(couple_ids)} couples")
    
    if not couple_ids:
        print("All couples already processed!")
        return 0
    
    # 세마포어로 동시 요청 제한
    semaphore = asyncio.Semaphore(config.max_concurrent)
    
    # 비동기 처리
    async with aiofiles.open(output_path, 'a', encoding='utf-8') as out_file:
        tasks = [process_couple(model, cid, data_dir, semaphore) for cid in couple_ids]
        
        results = []
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Labeling"):
            result = await coro
            if result:
                await out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                await out_file.flush()
                results.append(result)
    
    # 결과 요약
    print(f"\n{'='*60}")
    print(f"✅ Labeling Complete!")
    print(f"{'='*60}")
    print(f"Processed: {len(results)} couples")
    
    match_count = sum(1 for r in results if r.get('style_match'))
    print(f"\n📊 스타일 일치율: {match_count}/{len(results)} ({match_count/len(results)*100:.1f}%)")
    
    return 0


if __name__ == "__main__":
    asyncio.run(main())
