import re
from functools import lru_cache

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
nltk.download("wordnet")

MAX_CHARS = 5000    # only the first 5000 characters of each article are used
MIN_WORDS = 20      # drop articles shorter than this after cleaning

df = pd.read_csv("merged_raw.csv")
df["title"] = df["title"].fillna("").astype(str)
df["text"] = df["text"].fillna("").astype(str).str[:MAX_CHARS]

# Website names that would let the model cheat. Add more if you spot them.
SITE_NAMES = [
    "reuters", "abovetopsecret", "fellowshipoftheminds", "zerohedge",
    "beforeitsnews", "thedailysheeple", "infiniteunknown", "nytimes",
    "new york times", "theinternetpost", "shoebat", "newswithviews",
    "21st century wire", "breitbart", "infowars",
    "infinite unknown", "daily sheeple", "before it's news", "above top secret",
    "theodore shoebat", "walid shoebat", "christian post",
]
site_pattern = re.compile("|".join(SITE_NAMES), flags=re.I)


def strip_source_tags(text):
    # "WASHINGTON (Reuters) - " style opening
    text = re.sub(r"^.{0,100}?\(Reuters\)\s*-?\s*", "", text)
    # image credits, links, website names
    text = re.sub(r"featured image via [^.]*", "", text, flags=re.I)
    text = re.sub(r"(photo|image) (by|via|credit)[^.]*", "", text, flags=re.I)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"pic\.twitter\.com/\S*|\S*twitter\.com/\S*", "", text, flags=re.I)
    text = re.sub(r"continue reading the main story|reading the main story", "", text, flags=re.I)
    text = re.sub(r"\badvertisement\b|\bsupported by\b|expand\s*/?\s*collapse", "", text, flags=re.I)
    text = re.sub(r"\bDCG\b", "", text)
    text = re.sub(r"follow [^.]{0,100}(twitter|facebook)[^.]*", "", text, flags=re.I)
    text = site_pattern.sub("", text)
    return text


# 1. Remove giveaways from title and text separately, then combine
df["title"] = df["title"].apply(strip_source_tags)
df["text"] = df["text"].apply(strip_source_tags)
df["content"] = df["title"] + ". " + df["text"]

# 2. Light cleaning for the deep learning model
df["clean_dl"] = (df["content"].str.lower()
                  .str.replace(r"[^a-z\s]", " ", regex=True)
                  .str.replace(r"\s+", " ", regex=True)
                  .str.strip())

# 3. Heavier cleaning for Logistic Regression (stopwords out, words reduced to root)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


@lru_cache(maxsize=None)          # remembers results, so repeated words are fast
def lemma(word):
    return lemmatizer.lemmatize(word)


def to_ml(text):
    return " ".join(lemma(w) for w in text.split()
                    if w not in stop_words and len(w) > 2)


df["clean_ml"] = df["clean_dl"].apply(to_ml)

# 4. Drop very short articles
df = df[df["clean_dl"].str.split().str.len() >= MIN_WORDS]
print("After dropping short articles:", len(df))

# 5. Split 80 / 10 / 10, keeping the fake/real ratio equal for every source
df = df[["clean_ml", "clean_dl", "label", "source"]].reset_index(drop=True)
strat = df["label"].astype(str) + "_" + df["source"]

train, temp = train_test_split(df, test_size=0.2, random_state=42, stratify=strat)
val, test = train_test_split(temp, test_size=0.5, random_state=42,
                             stratify=temp["label"].astype(str) + "_" + temp["source"])

train.to_csv("data_train.csv", index=False)
val.to_csv("data_val.csv", index=False)
test.to_csv("data_test.csv", index=False)

print("\nSizes -> train:", len(train), "| val:", len(val), "| test:", len(test))
print("\nFake/real count by source (label 0 = fake, 1 = real):")
print(pd.crosstab(df["source"], df["label"]))
print("\nSample cleaned article:\n", train["clean_ml"].iloc[0][:300])
