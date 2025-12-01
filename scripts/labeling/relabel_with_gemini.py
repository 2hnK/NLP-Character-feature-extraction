import os
import json
import argparse
import time
import boto3
import io
from PIL import Image
from tqdm import tqdm
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

"""
Gemini API 기반 데이터셋 재라벨링 스크립트

이 스크립트는 S3에서 이미지를 다운로드하고, Google Gemini API를 사용하여
Fashion Style, Shot Type, Physical Features, Caption 등을 추출합니다.

주요 기능:
1. S3 이미지 로드 (boto3 사용)
2. Gemini API (gemini-3-pro-preview) 호출
3. 구조화된 JSON 메타데이터 생성 및 저장

사용법:
    export GOOGLE_API_KEY="your_api_key"
    python scripts/labeling/relabel_with_gemini.py --bucket sometimes-ki-datasets --input train_aug_final.jsonl --output train_aug_relabeled.jsonl
"""

# --- Configuration ---
DEFAULT_MODEL_NAME = "gemini-2.0-flash-exp"

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

def setup_gemini(api_key):
    genai.configure(api_key=api_key)

def get_gemini_response(model, image, prompt):
    try:
        response = model.generate_content(
            [prompt, image],
            generation_config={"response_mime_type": "application/json"},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error generating content: {e}")
        return None

def download_image_from_s3(s3_client, bucket, key):
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        image_data = obj['Body'].read()
        return Image.open(io.BytesIO(image_data)).convert('RGB')
    except Exception as e:
        print(f"Error downloading {key}: {e}")
        return None

def process_dataset(args):
    # 1. Setup Gemini
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return

    setup_gemini(api_key)
    model = genai.GenerativeModel(args.model_name)
    
    # 2. Setup S3
    try:
        s3 = boto3.client('s3')
        s3.list_buckets() # Test credentials
    except Exception as e:
        print(f"Error: AWS Credentials not found or invalid. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        print(f"Details: {e}")
        return
    
    # 3. Process
    print(f"Processing {args.input} -> {args.output}")
    print(f"Model: {args.model_name}")
    
    processed_count = 0
    error_count = 0
    
    # Check if output exists to resume? (Simple version: append mode)
    mode = 'a' if os.path.exists(args.output) else 'w'
    
    existing_ids = set()
    if mode == 'a':
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_ids.add(data.get('id') or data.get('filename')) # Use ID or filename as key
                except: pass
        print(f"Resuming... Found {len(existing_ids)} already processed items.")

    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, mode, encoding='utf-8') as fout:
        
        lines = fin.readlines()
        
        if args.limit:
            lines = lines[:args.limit]
            print(f"Limiting processing to first {args.limit} items.")
        
        for line in tqdm(lines, desc="Relabeling"):
            if not line.strip(): continue
            
            item = json.loads(line)
            
            # Identify ID for skipping
            item_id = item.get('id') or item.get('filename')
            if item_id in existing_ids:
                continue
                
            # Construct S3 Key
            # Logic from S3Dataset: prefix + filename
            # But here we might need to know the prefix.
            # Let's assume the input JSONL has 'image_path' (from align_paths.py) or we use args.prefix
            
            s3_key = None
            if 'image_path' in item:
                s3_key = item['image_path']
                # Remove bucket name if it's included (e.g., "bucket/path/file.jpg" -> "path/file.jpg")
                if s3_key.startswith(args.bucket + "/"):
                    s3_key = s3_key[len(args.bucket) + 1:]
            elif 'filename' in item:
                s3_key = os.path.join(args.prefix, item['filename']).replace("\\", "/")
            elif 'image_filename' in item:
                s3_key = os.path.join(args.prefix, item['image_filename']).replace("\\", "/")
            
            if not s3_key:
                print(f"Skipping item {item_id}: Could not determine S3 key.")
                error_count += 1
                continue
                
            # Download Image
            image = download_image_from_s3(s3, args.bucket, s3_key)
            if not image:
                error_count += 1
                continue
                
            # Inference
            result = get_gemini_response(model, image, SYSTEM_PROMPT)
            
            if result:
                # Create clean output with only necessary fields
                output_item = {
                    'id': item.get('id'),
                    'image_path': item.get('image_path'),
                    'image_filename': item.get('image_filename'),
                    'image_metadata': result
                }
                
                # Write to output
                fout.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                fout.flush() # Ensure write
                processed_count += 1
                
                # No sleep needed for Flash models
            else:
                print(f"Failed to process {s3_key}")
                error_count += 1
                
    print(f"\nDone! Processed: {processed_count}, Errors: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, default="sometimes-ki-datasets")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--prefix", type=str, default="", help="S3 prefix if image_path is missing")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items for testing")
    
    args = parser.parse_args()
    process_dataset(args)
