import os
import json
import argparse
import asyncio
import aiohttp
import aioboto3
import io
from PIL import Image
from tqdm.asyncio import tqdm as async_tqdm
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import List, Dict, Optional
import time
from dataclasses import dataclass
from dotenv import load_dotenv

# .env 파일 로드 (환경 변수 설정 편의성)
load_dotenv()

"""
Gemini API 기반 데이터셋 재라벨링 스크립트 (최적화 버전)

주요 최적화:
1. ✅ 비동기 처리 (asyncio): Gemini API 호출 및 S3 다운로드 병렬 처리
2. ✅ 배치 처리: 여러 이미지를 동시에 처리
3. ✅ Rate Limiting: API 제한 준수 (Semaphore 사용)
4. ✅ 재시도 로직: 실패한 요청 자동 재시도
5. ✅ 진행 상태 저장: 중단 시 이어서 처리 가능

성능 향상:
- 기존 대비 5-10배 빠른 처리 속도 (동시 요청 수에 따라)
- 기본 10개 동시 요청, 최대 50개까지 설정 가능

사용법:
    export GOOGLE_API_KEY="your_api_key"
    python scripts/labeling/relabel_with_gemini_optimized.py \
        --bucket sometimes-ki-datasets \
        --input train_aug_final.jsonl \
        --output train_aug_relabeled.jsonl \
        --concurrent 20 \
        --model-name gemini-2.5-flash
"""

# --- Configuration ---
DEFAULT_MODEL_NAME = "gemini-2.5-flash"  # 속도 최적화를 위해 Flash 모델로 변경
DEFAULT_CONCURRENT_REQUESTS = 10  # 동시 요청 수
MAX_RETRIES = 3  # 재시도 횟수
RETRY_DELAY = 1.0  # 재시도 대기 시간 (초)

SYSTEM_PROMPT = """
Role: Expert Fashion & Social Profile Analyst for Dating Apps.
Output Language: ENGLISH ONLY.

Task: Extract structured metadata from the profile image.

--- 1. CATEGORICAL FIELDS (Strict Enums) ---
Choose exactly ONE value for each category based on visual evidence.

A. fashion_style:
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
{
  "fashion_style": "Enum Value",
  "shot_type": "Enum Value",
  "visual_quality": "Enum Value",
  "physical_features": ["trait1", "trait2", ...],
  "caption": "Full sentence description."
}
"""


@dataclass
class ProcessingStats:
    """처리 통계"""
    total: int = 0
    processed: int = 0
    errors: int = 0
    skipped: int = 0
    
    def __str__(self):
        return f"Total: {self.total}, Processed: {self.processed}, Errors: {self.errors}, Skipped: {self.skipped}"


class GeminiProcessor:
    """비동기 Gemini API 프로세서"""
    
    def __init__(self, model_name: str, api_key: str, max_concurrent: int = 10):
        self.model_name = model_name
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Gemini 설정
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    async def generate_content(self, image: Image.Image, prompt: str, retry: int = 0) -> Optional[Dict]:
        """비동기 Gemini API 호출 (재시도 로직 포함)"""
        async with self.semaphore:  # Rate Limiting
            try:
                # Gemini SDK는 동기 함수이므로 executor에서 실행
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate_content(
                        [prompt, image],
                        generation_config={"response_mime_type": "application/json"},
                        request_options={"timeout": 600},  # 600초(10분) 타임아웃 설정
                        safety_settings={
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        }
                    )
                )
                
                return json.loads(response.text)
                
            except Exception as e:
                if retry < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * (2 ** retry))  # Exponential backoff
                    return await self.generate_content(image, prompt, retry + 1)
                else:
                    print(f"Error after {MAX_RETRIES} retries: {e}")
                    return None


class S3ImageLoader:
    """비동기 S3 이미지 로더"""
    
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.session = aioboto3.Session()
        
    async def download_image(self, key: str) -> Optional[Image.Image]:
        """S3에서 이미지 비동기 다운로드"""
        try:
            async with self.session.client('s3') as s3:
                response = await s3.get_object(Bucket=self.bucket, Key=key)
                async with response['Body'] as stream:
                    image_data = await stream.read()
                
                # 이미지 디코딩 (CPU 집약적 작업이므로 executor 사용)
                loop = asyncio.get_event_loop()
                image = await loop.run_in_executor(
                    None,
                    lambda: Image.open(io.BytesIO(image_data)).convert('RGB')
                )
                return image
                
        except Exception as e:
            print(f"Error downloading {key}: {e}")
            return None

    async def list_images(self, prefix: str) -> List[Dict]:
        """S3 Prefix 내의 모든 이미지 파일 목록 조회"""
        items = []
        try:
            async with self.session.client('s3') as s3:
                paginator = s3.get_paginator('list_objects_v2')
                async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                    if 'Contents' not in page:
                        continue
                        
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                            # 파일명에서 ID 추출 (예: aug_00001.jpg -> aug_00001)
                            filename = os.path.basename(key)
                            file_id = os.path.splitext(filename)[0]
                            
                            items.append({
                                'id': file_id,
                                'filename': filename,
                                'image_path': key  # 전체 키 저장
                            })
            print(f"Found {len(items)} images in s3://{self.bucket}/{prefix}")
            return items
            
        except Exception as e:
            print(f"Error listing objects: {e}")
            return []


async def process_single_item(
    item: Dict,
    bucket: str,
    prefix: str,
    s3_loader: S3ImageLoader,
    gemini_processor: GeminiProcessor,
    existing_ids: set
) -> Optional[Dict]:
    """단일 아이템 비동기 처리"""
    
    # 아이템 ID 확인
    item_id = item.get('id') or item.get('filename')
    if item_id in existing_ids:
        return None  # 이미 처리됨
    
    # S3 키 결정
    s3_key = item.get('image_path')
    
    # JSONL 모드일 경우 경로 보정
    if not s3_key:
        if 'filename' in item:
            s3_key = os.path.join(prefix, item['filename']).replace("\\", "/")
        elif 'image_filename' in item:
            s3_key = os.path.join(prefix, item['image_filename']).replace("\\", "/")
            
    # 경로가 bucket 이름으로 시작하면 제거
    if s3_key and s3_key.startswith(bucket + "/"):
        s3_key = s3_key[len(bucket) + 1:]
    
    if not s3_key:
        return {'error': f"Could not determine S3 key for {item_id}"}
    
    # 이미지 다운로드
    image = await s3_loader.download_image(s3_key)
    if not image:
        return {'error': f"Failed to download {s3_key}"}
    
    # Gemini API 호출
    result = await gemini_processor.generate_content(image, SYSTEM_PROMPT)
    if not result:
        return {'error': f"Failed to process {s3_key}"}
    
    # 결과 반환
    return {
        'id': item.get('id'),
        'image_path': s3_key,
        'image_filename': os.path.basename(s3_key),
        'image_metadata': result
    }


async def process_batch(
    items: List[Dict],
    args: argparse.Namespace,
    s3_loader: S3ImageLoader,
    gemini_processor: GeminiProcessor,
    existing_ids: set,
    output_file
) -> ProcessingStats:
    """배치 비동기 처리"""
    stats = ProcessingStats(total=len(items))
    
    # 모든 아이템을 비동기로 처리
    tasks = [
        process_single_item(item, args.bucket, args.prefix, s3_loader, gemini_processor, existing_ids)
        for item in items
    ]
    
    # 진행 상태 표시와 함께 실행
    results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
        result = await coro
        results.append(result)
    
    # 결과 저장
    for result in results:
        if result is None:
            stats.skipped += 1
        elif 'error' in result:
            stats.errors += 1
        else:
            output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            output_file.flush()
            stats.processed += 1
    
    return stats


async def process_dataset_async(args: argparse.Namespace):
    """비동기 데이터셋 처리 메인 함수"""
    
    # 1. API 키 확인
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return
    
    # 2. 초기화
    print(f"Initializing with model: {args.model_name}")
    print(f"Concurrent requests: {args.concurrent}")
    print(f"Bucket: {args.bucket}")
    print(f"Mode: {args.mode}")
    
    gemini_processor = GeminiProcessor(args.model_name, api_key, args.concurrent)
    s3_loader = S3ImageLoader(args.bucket)
    
    # 3. 기존 처리 항목 로드
    existing_ids = set()
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_ids.add(data.get('id') or data.get('filename') or data.get('image_filename'))
                except:
                    pass
        print(f"Found {len(existing_ids)} already processed items. Resuming...")
    
    # 4. 입력 데이터 로드 (모드별 분기)
    items = []
    if args.mode == 's3_folder':
        print(f"Scanning S3 folder: {args.prefix}...")
        items = await s3_loader.list_images(args.prefix)
    else:
        # JSONL 모드
        if not args.input:
            print("Error: --input argument is required for jsonl mode.")
            return
        print(f"Reading input file: {args.input}...")
        with open(args.input, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        items = [json.loads(line) for line in lines if line.strip()]
    
    if not items:
        print("No items found to process.")
        return

    if args.limit:
        items = items[:args.limit]
        print(f"Limiting to first {args.limit} items.")
    
    print(f"Total items to process: {len(items)}")
    
    # 5. 배치 처리
    mode = 'a' if os.path.exists(args.output) else 'w'
    overall_stats = ProcessingStats(total=len(items))
    
    start_time = time.time()
    
    with open(args.output, mode, encoding='utf-8') as output_file:
        # 배치 크기 설정 (메모리 관리)
        batch_size = args.batch_size or args.concurrent * 5
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            print(f"\n=== Batch {i // batch_size + 1}/{(len(items) + batch_size - 1) // batch_size} ===")
            
            batch_stats = await process_batch(
                batch, args, s3_loader, gemini_processor, existing_ids, output_file
            )
            
            # 통계 업데이트
            overall_stats.processed += batch_stats.processed
            overall_stats.errors += batch_stats.errors
            overall_stats.skipped += batch_stats.skipped
            
            print(f"Batch stats: {batch_stats}")
    
    # 6. 최종 결과
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Processing Complete!")
    print(f"{'='*60}")
    print(f"Overall: {overall_stats}")
    print(f"Time elapsed: {elapsed:.2f}s")
    if elapsed > 0:
        print(f"Speed: {overall_stats.processed / elapsed:.2f} items/sec")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Optimized Gemini-based dataset relabeling")
    parser.add_argument("--mode", type=str, default="jsonl", choices=["jsonl", "s3_folder"],
                        help="Processing mode: 'jsonl' (from file) or 's3_folder' (scan bucket)")
    parser.add_argument("--bucket", type=str, default="sometimes-ki-datasets", help="S3 bucket name")
    parser.add_argument("--input", type=str, help="Input JSONL file (required for jsonl mode)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--prefix", type=str, default="", help="S3 prefix (folder path) to scan or prepend")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Gemini model name")
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT_REQUESTS, 
                        help="Maximum concurrent API requests")
    parser.add_argument("--batch-size", type=int, default=None, 
                        help="Batch size for processing (default: concurrent * 5)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items for testing")
    
    args = parser.parse_args()
    
    # 비동기 실행
    asyncio.run(process_dataset_async(args))


if __name__ == "__main__":
    main()
