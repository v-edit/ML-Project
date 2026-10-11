import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, recall_score)
from sklearn.utils.class_weight import compute_class_weight

keras.utils.set_random_seed(42)

MAX_TOKENS = 50000   # vocabulary size: the 50,000 most common words
SEQ_LEN = 300        # each article is cut or padded to 300 words
EMBED_DIM = 128      # each word becomes a list of 128 numbers
BATCH = 128
EPOCHS = 6

print("GPU found:", tf.config.list_physical_devices("GPU"))

train = pd.read_csv("dl_train.csv")
val = pd.read_csv("dl_val.csv")
for d in (train, val):
    d["clean_dl"] = d["clean_dl"].fillna("").astype(str)

# 1. Class weights, so the smaller class (real) is not ignored
cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=train["label"])
class_weight = {0: float(cw[0]), 1: float(cw[1])}
print("Class weights:", class_weight)

# 2. Turn words into numbers. The vocabulary is built from TRAIN only
vectorizer = layers.TextVectorization(max_tokens=MAX_TOKENS, output_sequence_length=SEQ_LEN)
vectorizer.adapt(train["clean_dl"].tolist())


def make_ds(df, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((df["clean_dl"].tolist(),
                                             df["label"].values.astype("float32")))
    if shuffle:
        ds = ds.shuffle(10000, seed=42)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(train, shuffle=True)
val_ds = make_ds(val)

# 3. The model
inputs = keras.Input(shape=(), dtype="string")
x = vectorizer(inputs)                                   # words -> numbers
x = layers.Embedding(MAX_TOKENS, EMBED_DIM, mask_zero=True)(x)   # numbers -> meaning vectors
x = layers.Bidirectional(layers.LSTM(64))(x)             # reads the article both ways
x = layers.Dropout(0.4)(x)                               # randomly switches off units to avoid memorising
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)       # probability that the article is real
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
model.summary()

# 4. Train; stop early if validation loss stops improving
early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                      restore_best_weights=True)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
          class_weight=class_weight, callbacks=[early])

# 5. Evaluate on validation
probs = model.predict(val_ds).ravel()
pred = (probs >= 0.5).astype(int)
y_val = val["label"].values

print("\nAccuracy:", round(accuracy_score(y_val, pred), 4))
print(classification_report(y_val, pred, target_names=["fake (0)", "real (1)"], digits=4))
print("Confusion matrix (rows = true, columns = predicted):")
print(confusion_matrix(y_val, pred))

print("\nPer-source results on validation:")
for s in sorted(val["source"].unique()):
    m = (val["source"] == s).values
    print(f"{s:12s} accuracy = {accuracy_score(y_val[m], pred[m]):.4f} | "
          f"fake recall = {recall_score(y_val[m], pred[m], pos_label=0):.4f} | "
          f"real recall = {recall_score(y_val[m], pred[m], pos_label=1):.4f}")

# 6. Save the model (the word-to-number step is saved inside it)
model.save("bilstm_model.keras")
print("\nSaved bilstm_model.keras")
