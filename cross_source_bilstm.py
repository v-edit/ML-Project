import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

MAX_TOKENS = 50000
SEQ_LEN = 300
EMBED_DIM = 128
BATCH = 128
EPOCHS = 4          # early stopping will usually stop sooner

print("GPU found:", tf.config.list_physical_devices("GPU"))

# Only train + validation rows are used here. The test set stays untouched.
allrows = pd.concat([pd.read_csv("dl_train.csv"), pd.read_csv("dl_val.csv")],
                    ignore_index=True)
allrows["clean_dl"] = allrows["clean_dl"].fillna("").astype(str)


def make_ds(df, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((df["clean_dl"].tolist(),
                                             df["label"].values.astype("float32")))
    if shuffle:
        ds = ds.shuffle(10000, seed=42)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


def build_model(vectorizer):
    inputs = keras.Input(shape=(), dtype="string")
    x = vectorizer(inputs)
    x = layers.Embedding(MAX_TOKENS, EMBED_DIM, mask_zero=True)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


results = []
for held_out in sorted(allrows["source"].unique()):
    print(f"\n===== Held-out source: {held_out} =====")
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)

    seen = allrows[allrows["source"] != held_out]
    unseen = allrows[allrows["source"] == held_out]

    # A small slice of the SEEN sources is used for early stopping,
    # so the held-out source is never used to pick the best epoch.
    fit, stop = train_test_split(
        seen, test_size=0.1, random_state=42,
        stratify=seen["label"].astype(str) + "_" + seen["source"])

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=fit["label"])
    class_weight = {0: float(cw[0]), 1: float(cw[1])}

    vectorizer = layers.TextVectorization(max_tokens=MAX_TOKENS,
                                          output_sequence_length=SEQ_LEN)
    vectorizer.adapt(fit["clean_dl"].tolist())     # vocabulary from training rows only

    model = build_model(vectorizer)
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=1,
                                          restore_best_weights=True)
    model.fit(make_ds(fit, shuffle=True), validation_data=make_ds(stop),
              epochs=EPOCHS, class_weight=class_weight, callbacks=[early], verbose=2)

    probs = model.predict(make_ds(unseen), verbose=0).ravel()
    pred = (probs >= 0.5).astype(int)
    y = unseen["label"].values
    row = {
        "held_out": held_out,
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "fake_recall": recall_score(y, pred, pos_label=0),
        "real_recall": recall_score(y, pred, pos_label=1),
    }
    results.append(row)
    print({k: (round(v, 4) if not isinstance(v, str) else v) for k, v in row.items()})

print("\n===== Bi-LSTM, trained on two sources and tested on the third =====")
print(pd.DataFrame(results).round(4).to_string(index=False))
print("\nLogistic Regression for comparison (from your earlier run):")
print("kaggle      accuracy 0.7460 | macro-F1 0.7452")
print("opensources accuracy 0.8392 | macro-F1 0.6918")
print("welfake     accuracy 0.6584 | macro-F1 0.6541")
