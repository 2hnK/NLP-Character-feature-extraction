import asyncio
import aioboto3
import io
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

async def check_images():
    bucket = 'sometimes-ki-datasets'
    prefix = 'dataset/validation/images/'
    
    session = aioboto3.Session()
    print(f"Scanning s3://{bucket}/{prefix} for corrupt images...")
    
    corrupt_files = []
    checked_count = 0
    
    async with session.client('s3') as s3:
        paginator = s3.get_paginator('list_objects_v2')
        async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                if not key.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    continue
                    
                checked_count += 1
                if checked_count % 10 == 0:
                    print(f"Checked {checked_count} images...", end='\r')
                
                try:
                    response = await s3.get_object(Bucket=bucket, Key=key)
                    image_data = await response['Body'].read()
                    
                    try:
                        Image.open(io.BytesIO(image_data)).verify()
                    except Exception as e:
                        print(f"\n[CORRUPT] {key}: {e}")
                        corrupt_files.append(key)
                        
                except Exception as e:
                    print(f"\n[ERROR] Failed to download {key}: {e}")
                    corrupt_files.append(key)

    print(f"\n\nScan complete. Checked {checked_count} images.")
    if corrupt_files:
        print(f"Found {len(corrupt_files)} corrupt files:")
        for f in corrupt_files:
            print(f" - {f}")
    else:
        print("No corrupt files found.")

if __name__ == "__main__":
    asyncio.run(check_images())
