import json

def inspect_file(filepath):
    print(f"--- Inspecting {filepath} ---")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line:
                print("File is empty")
                return
            data = json.loads(first_line)
            print("Keys:", list(data.keys()))
            print("Sample Data:", json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

inspect_file('data/positive_pairs.jsonl')
inspect_file('data/negative_pairs.jsonl')
