import pandas as pd
from collections import Counter
from urllib.parse import urlparse
from datasets import load_dataset

TARGET_FAKE = 55000         # stop once we have this many fake articles
MAX_PER_DOMAIN = 3000       # no single website can contribute more than this
SCAN_LIMIT = 800_000        # give up after reading this many rows
MAX_CHARS = 5000            # keep only the first 5000 characters of each article
FAKE_TYPES = {"fake", "conspiracy"}
OUT = "opensources_sample.csv"

d = load_dataset("andyP/fake_news_en_opensources", streaming=True)
ds = d["train"] if "train" in d else next(iter(d.values()))
ds = ds.shuffle(seed=42, buffer_size=10000)

rows = []
counts = {0: 0, 1: 0}
per_domain = Counter()


def save():
    pd.DataFrame(rows).to_csv(OUT, index=False)   # saved often, so a crash loses little


for i, r in enumerate(ds):
    if i > 0 and i % 100000 == 0:
        print(f"scanned {i:,} rows | fake kept: {counts[0]:,} | real kept: {counts[1]:,}")
        save()
    if i >= SCAN_LIMIT or counts[0] >= TARGET_FAKE:
        break

    t = r["type"]
    if t == "reliable":
        label = 1                     # real articles are rare early in the file; keep any we meet
    elif t in FAKE_TYPES:
        label = 0
    else:
        continue                      # skip satire, bias, unreliable, etc.

    text = r["content"]
    if not text or not text.strip():
        continue

    domain = r.get("domain") or urlparse(r.get("url") or "").netloc.lower().replace("www.", "")
    if per_domain[domain] >= MAX_PER_DOMAIN:
        continue

    per_domain[domain] += 1
    counts[label] += 1
    rows.append({"title": r["title"], "text": text[:MAX_CHARS],
                 "label": label, "domain": domain})

save()
df = pd.DataFrame(rows)
print("\nRows kept:", len(df))
print(df["label"].value_counts())
print("Domains per class:\n", df.groupby("label")["domain"].nunique())
print("Top 10 domains:\n", df["domain"].value_counts().head(10))