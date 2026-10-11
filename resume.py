import pandas as pd
from collections import Counter
from urllib.parse import urlparse
from datasets import load_dataset

RESUME_FROM = 400_000       # the LAST "scanned N rows" number you saw before stopping
PART1_FILE = "opensources_part1.csv"

TARGET_FAKE = 55000         # stop once we have this many fake articles
MAX_PER_DOMAIN = 3000       # no single website can contribute more than this
SCAN_LIMIT = 800_000        # give up after reading this many rows in total
MAX_CHARS = 5000
FAKE_TYPES = {"fake", "conspiracy"}
OUT = "opensources_sample.csv"

# 1. Load what we already collected
try:
    old = pd.read_csv(PART1_FILE)
except UnicodeDecodeError:
    # file was probably re-saved by Excel; fall back to the Windows encoding
    old = pd.read_csv(PART1_FILE, encoding="cp1252", encoding_errors="replace")
rows = old.to_dict("records")
counts = {0: int((old["label"] == 0).sum()), 1: int((old["label"] == 1).sum())}
per_domain = Counter(old["domain"].fillna(""))
seen = set(zip(old["title"].fillna("").astype(str).str[:80],
               old["text"].fillna("").astype(str).str[:80]))   # safety net against repeats
print(f"Loaded {len(old):,} rows | fake: {counts[0]:,} | real: {counts[1]:,}")

# 2. Same stream and same shuffle seed as before, then skip the rows already read
d = load_dataset("andyP/fake_news_en_opensources", streaming=True)
ds = d["train"] if "train" in d else next(iter(d.values()))
ds = ds.shuffle(seed=42, buffer_size=10000)
ds = ds.skip(RESUME_FROM)   # still has to read past them, so this part takes a while
print(f"Skipping the first {RESUME_FROM:,} rows (already collected)...")


def save():
    pd.DataFrame(rows).to_csv(OUT, index=False)


# 3. Continue collecting
for i, r in enumerate(ds, start=RESUME_FROM):
    if i > RESUME_FROM and i % 100000 == 0:
        print(f"scanned {i:,} rows | fake kept: {counts[0]:,} | real kept: {counts[1]:,}")
        save()
    if i >= SCAN_LIMIT or counts[0] >= TARGET_FAKE:
        break

    t = r["type"]
    if t == "reliable":
        label = 1
    elif t in FAKE_TYPES:
        label = 0
    else:
        continue

    text = r["content"]
    if not text or not text.strip():
        continue

    key = (str(r["title"] or "")[:80], text[:80])
    if key in seen:
        continue

    domain = r.get("domain") or urlparse(r.get("url") or "").netloc.lower().replace("www.", "")
    if per_domain[domain] >= MAX_PER_DOMAIN:
        continue

    seen.add(key)
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