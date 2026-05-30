import json
from datasets import load_dataset
import os

print("Downloading full HuffPost dataset (209,527 articles) from HuggingFace...")
ds = load_dataset("heegyu/news-category-dataset", split="train")

records = []
for row in ds:
    # convert date to string if it's not
    date_str = str(row['date'])[:10] if row['date'] else ""
    records.append({
        "link": row["link"] or "",
        "headline": row["headline"] or "",
        "category": row["category"] or "UNKNOWN",
        "short_description": row["short_description"] or "",
        "authors": row["authors"] or "",
        "date": date_str
    })

out_path = "data/raw/huffpost_news_full.json"
with open(out_path, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

size_mb = os.path.getsize(out_path) / (1024 * 1024)
print(f"Full dataset saved to {out_path} ({size_mb:.1f} MB)!")
