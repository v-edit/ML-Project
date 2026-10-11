import pandas as pd

MAX_WORDS = 400   # the model only reads the first 300 words anyway

for name in ["train", "val", "test"]:
    df = pd.read_csv(f"data_{name}.csv", usecols=["clean_dl", "label", "source"])
    df["clean_dl"] = df["clean_dl"].fillna("").apply(lambda t: " ".join(t.split()[:MAX_WORDS]))
    df.to_csv(f"dl_{name}.csv", index=False)
    print(name, len(df), "rows saved to", f"dl_{name}.csv")
