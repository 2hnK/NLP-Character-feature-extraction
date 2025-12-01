import asyncio
import aioboto3
import os
from dotenv import load_dotenv

load_dotenv()

async def check_manifest_count():
    bucket = 'sometimes-ki-datasets'
    key = 'dataset/qwen-vl-train-v1/train_aug_final.jsonl'
    
    session = aioboto3.Session()
    print(f"Reading s3://{bucket}/{key}...")
    
    try:
        async with session.client('s3') as s3:
            response = await s3.get_object(Bucket=bucket, Key=key)
            content = await response['Body'].read()
            lines = content.decode('utf-8').strip().split('\n')
            print(f"Total lines in {key}: {len(lines)}")
            
            # Optional: Check for missing images
            # This would require listing images again, which we just did.
            # For now, just confirming the count is enough.
            
    except Exception as e:
        print(f"Error reading manifest: {e}")

if __name__ == "__main__":
    asyncio.run(check_manifest_count())
