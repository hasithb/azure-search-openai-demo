import json
from collections import defaultdict

with open('data/legal-scraper/processed/v3_full_corrected.json', encoding='utf-8') as f:
    docs = json.load(f)

url_to_sections = defaultdict(list)
for d in docs:
    url = d['storageUrl']
    subs = set(d.get('subsections') or [])
    if d.get('subsection_id'):
        subs.add(d['subsection_id'])
    url_to_sections[url].extend(list(subs))

result = {}
for url, sections in url_to_sections.items():
    def sort_key(s):
        try:
            parts = s.split('.')
            return [float(p) if p.isdigit() else p for p in parts]
        except:
            return [str(s)]
            
    # Sort using a custom key that handles mixed types safely
    def safe_sort_key(s):
        parts = s.split('.')
        res = []
        for p in parts:
            if p.isdigit():
                res.append((0, float(p)))
            else:
                res.append((1, p))
        return res
        
    result[url] = sorted(list(set(sections)), key=safe_sort_key)

output_path = 'data/legal-scraper/processed/all_sections_by_url.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)

print(f"Generated {output_path} with {len(result)} URLs.")
sample_url = list(result.keys())[0]
print(f"\nSample for {sample_url}:")
print(json.dumps(result[sample_url], indent=2))
