import json
import argparse
import os
from pathlib import Path

"""
JSONL 전처리 및 텍스트 프롬프트 생성 스크립트

이 스크립트는 초기 JSONL 파일을 처리하여 다음을 수행합니다:
1. 각 항목에 순차적인 파일명(`aug_XXXXX.jpg`) 할당.
2. 메타데이터를 기반으로 Qwen 모델 입력을 위한 구조화된 텍스트(`text_input`) 생성.
3. 라벨 매핑 파일 생성.

사용법:
    python preprocess_jsonl.py --input <input_file> --output <output_file>
"""

def preprocess_jsonl(input_path, output_path, mapping_output_path, start_index=0):
    """
    Reads a JSONL file, adds a 'filename' field to each entry sequentially,
    formats text fields for Qwen input, and generates a label mapping.
    
    Filename format: aug_{index:05d}.jpg
    Text format: "Style: {style}. Features: {features}. Vibe: {vibe}."
    """
    print(f"Processing {input_path} -> {output_path}")
    
    style_counts = {}
    styles = set()
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for i, line in enumerate(infile):
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # 1. Add filename
            idx = start_index + i
            filename = f"aug_{idx:05d}.jpg"
            data['filename'] = filename
            
            # 2. Extract and Count Styles
            if 'image_metadata' in data and 'fashion_style' in data['image_metadata']:
                style = data['image_metadata']['fashion_style']
                styles.add(style)
                style_counts[style] = style_counts.get(style, 0) + 1
            
            # 3. Format Text Input
            # "Style: Sporty_Athleisure. Features: Short dark hair, Clean-shaven face. Vibe: Energetic_Active."
            if 'image_metadata' in data:
                meta = data['image_metadata']
                style = meta.get('fashion_style', 'Unknown')
                vibe = meta.get('vibe_category', 'Unknown')
                features = meta.get('physical_features', [])
                
                if isinstance(features, list):
                    features_str = ", ".join(features)
                else:
                    features_str = str(features)
                    
                caption = meta.get('caption', '')
                
                # Combine into a structured prompt or caption
                # We store it in 'text_input' field
                text_input = f"Style: {style}. Features: {features_str}. Vibe: {vibe}."
                if caption:
                     text_input += f" Caption: {caption}"
                
                data['text_input'] = text_input
            
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            
    print(f"Processed {i+1} items.")
    print("\nStyle Distribution:")
    for style, count in style_counts.items():
        print(f"  {style}: {count}")
        
    # 4. Save Label Mapping
    # Sort styles to ensure deterministic mapping
    sorted_styles = sorted(list(styles))
    label_mapping = {style: idx for idx, style in enumerate(sorted_styles)}
    
    with open(mapping_output_path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=4, ensure_ascii=False)
        
    print(f"\nLabel mapping saved to {mapping_output_path}")
    print(f"Total classes: {len(label_mapping)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess JSONL for Triplet Loss Training")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--mapping_output", type=str, default="label_mapping.json", help="Output path for label mapping JSON")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index for filenames")
    
    args = parser.parse_args()
    
    preprocess_jsonl(args.input, args.output, args.mapping_output, args.start_index)
