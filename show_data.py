# show_data.py
import json
c = json.load(open('data/chunks.json', encoding='utf-8'))
print(f"Chunks: {len(c)}")
print(f"Sample company: {c[0]['company']}")
print(f"Sample text: {c[0]['text'][:150]}")