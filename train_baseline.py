import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)

RUN_CROSS_SOURCE = True   # the "unseen website" test; takes a few extra minutes

train = pd.read_csv("data_train.csv")
val = pd.read_csv("data_val.csv")
for d in (train, val):
    d["clean_ml"] = d["clean_ml"].fillna("")


def make_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9,
                           max_features=300000, sublinear_tf=True)


# 1. TF-IDF: fit on TRAIN only, then apply the same vocabulary to validation
vec = make_vectorizer()
X_train = vec.fit_transform(train["clean_ml"])
X_val = vec.transform(val["clean_ml"])
y_train, y_val = train["label"], val["label"]
print("Feature matrix:", X_train.shape)

# 2. Try a few values of C (how strongly the model is kept from over-fitting)
best = None
for C in [8, 20, 50]:
    model = LogisticRegression(C=C, class_weight="balanced",
                               solver="liblinear", max_iter=1000)
    model.fit(X_train, y_train)
    f1 = f1_score(y_val, model.predict(X_val), average="macro")
    print(f"C = {C}: validation macro-F1 = {f1:.4f}")
    if best is None or f1 > best[0]:
        best = (f1, C, model)

_, best_C, model = best
print("\nBest C:", best_C)

# 3. Full validation report
pred = model.predict(X_val)
print("\nAccuracy:", round(accuracy_score(y_val, pred), 4))
print(classification_report(y_val, pred, target_names=["fake (0)", "real (1)"], digits=4))
print("Confusion matrix (rows = true, columns = predicted):")
print(confusion_matrix(y_val, pred))

# 4. Results per source
print("\nPer-source results on validation:")
for s in sorted(val["source"].unique()):
    m = (val["source"] == s).values
    print(f"{s:12s} accuracy = {accuracy_score(y_val[m], pred[m]):.4f} | "
          f"fake recall = {recall_score(y_val[m], pred[m], pos_label=0):.4f} | "
          f"real recall = {recall_score(y_val[m], pred[m], pos_label=1):.4f}")

# 5. Which words push the prediction? (a quick check for cheating)
names = vec.get_feature_names_out()
order = model.coef_[0].argsort()
print("\nTop 25 words pushing toward FAKE:", [names[i] for i in order[:25]])
print("Top 25 words pushing toward REAL:", [names[i] for i in order[-25:][::-1]])

# 6. Save the model and its vectorizer together
joblib.dump({"vectorizer": vec, "model": model}, "baseline_lr.joblib")

# 7. Unseen-website test: train on two sources, test on the third
if RUN_CROSS_SOURCE:
    allrows = pd.concat([train, val], ignore_index=True)
    print("\nTrain on two sources, test on the third:")
    for s in sorted(allrows["source"].unique()):
        tr = allrows[allrows["source"] != s]
        te = allrows[allrows["source"] == s]
        v = make_vectorizer()
        Xtr = v.fit_transform(tr["clean_ml"])
        Xte = v.transform(te["clean_ml"])
        m2 = LogisticRegression(C=best_C, class_weight="balanced",
                                solver="liblinear", max_iter=1000).fit(Xtr, tr["label"])
        p = m2.predict(Xte)
        print(f"held-out {s:12s} accuracy = {accuracy_score(te['label'], p):.4f} | "
              f"macro-F1 = {f1_score(te['label'], p, average='macro'):.4f} | "
              f"fake recall = {recall_score(te['label'], p, pos_label=0):.4f} | "
              f"real recall = {recall_score(te['label'], p, pos_label=1):.4f}")