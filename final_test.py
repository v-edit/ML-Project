import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             recall_score, roc_auc_score, roc_curve)


def evaluate(name, y, probs_real, sources):
    pred = (probs_real >= 0.5).astype(int)
    print(f"\n================ {name} (TEST SET) ================")
    print("Accuracy:", round(accuracy_score(y, pred), 4))
    print("ROC AUC :", round(roc_auc_score(y, probs_real), 4))
    print(classification_report(y, pred, target_names=["fake (0)", "real (1)"], digits=4))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y, pred))

    print("\nPer-source results:")
    for s in sorted(set(sources)):
        m = np.array(sources) == s
        print(f"{s:12s} accuracy = {accuracy_score(y[m], pred[m]):.4f} | "
              f"fake recall = {recall_score(y[m], pred[m], pos_label=0):.4f} | "
              f"real recall = {recall_score(y[m], pred[m], pos_label=1):.4f}")

    # Plots for the report
    ConfusionMatrixDisplay(confusion_matrix(y, pred),
                           display_labels=["Fake", "Real"]).plot(cmap="Blues")
    plt.title(f"Confusion matrix - {name}")
    plt.savefig(f"confusion_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(y, probs_real)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y, probs_real):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve - {name}")
    plt.legend()
    plt.savefig(f"roc_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    pd.DataFrame({"label": y, "source": sources, "prob_real": probs_real}) \
        .to_csv(f"test_probs_{name}.csv", index=False)
    print(f"Saved confusion_{name}.png, roc_{name}.png, test_probs_{name}.csv")


ran_any = False

# --- Logistic Regression (needs baseline_lr.joblib and data_test.csv) ---
if os.path.exists("baseline_lr.joblib") and os.path.exists("data_test.csv"):
    bundle = joblib.load("baseline_lr.joblib")
    test = pd.read_csv("data_test.csv")
    X = bundle["vectorizer"].transform(test["clean_ml"].fillna(""))
    probs = bundle["model"].predict_proba(X)[:, 1]
    evaluate("LogisticRegression", test["label"].values, probs, test["source"].tolist())
    ran_any = True
else:
    print("Skipping Logistic Regression (baseline_lr.joblib or data_test.csv not found here)")

# --- Bi-LSTM (needs bilstm_model.keras and dl_test.csv; run this part on Colab) ---
if os.path.exists("bilstm_model.keras") and os.path.exists("dl_test.csv"):
    import tensorflow as tf
    import keras
    model = keras.models.load_model("bilstm_model.keras")
    test = pd.read_csv("dl_test.csv")
    texts = test["clean_dl"].fillna("").astype(str).tolist()
    ds = tf.data.Dataset.from_tensor_slices(texts).batch(256)
    probs = model.predict(ds, verbose=0).ravel()
    evaluate("BiLSTM", test["label"].values, probs, test["source"].tolist())
    ran_any = True
else:
    print("Skipping Bi-LSTM (bilstm_model.keras or dl_test.csv not found here)")

if not ran_any:
    print("\nNothing was evaluated. Put the model files and test CSVs in this folder.")
