# train.py
import os
import copy
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ——— CONFIG ———
CSV_PATH       = "training.csv"
MODEL_DIR      = os.path.join("model", "keypoint_classifier")
MODEL_FILENAME = "asl_model.keras"      
LABEL_FILENAME = "label.csv"

os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
LABEL_PATH = os.path.join(MODEL_DIR, LABEL_FILENAME)

# ——— your app.py’s preprocessing ———
def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    # shift so first landmark is at (0,0)
    base_x, base_y = temp[0][0], temp[0][1]
    for p in temp:
        p[0] -= base_x
        p[1] -= base_y
    flat = list(itertools.chain.from_iterable(temp))
    m = max(map(abs, flat)) or 1.0
    return [v / m for v in flat]

# ——— 1) LOAD & SPLIT RAW KEYPOINTS + LABELS ———
# raw CSV has no header, 64 cols: [label, x0,y0,z0, x1,y1,z1, … x20,y20,z20]
df = pd.read_csv(CSV_PATH, header=None)
y_raw = df.iloc[:, 0].astype(int).values           # 0–25
kp_raw = df.iloc[:, 1:].values.astype(np.float32)  # shape (N,63)

# ——— 2) APPLY SAME calc_landmark_list + pre_process_landmark ———
#    i.e. drop z, make relative, normalize
processed = []
for row in kp_raw:
    pts = row.reshape(-1, 3)               # (21,3)
    # build landmark_list = [[x,y],…]
    landmark_list = [[x, y] for x, y, z in pts]
    proc = pre_process_landmark(landmark_list)  # length 42
    processed.append(proc)
X = np.vstack(processed)  # shape (N,42)

# ——— 3) ENCODE CLASSES & WRITE label.csv ———
le = LabelEncoder()
y = le.fit_transform(y_raw)       # ensures 0..25 in sorted order
n_classes = len(le.classes_)      # should be 26

# we want A,B,C… for labels
letters = [chr(65 + i) for i in range(n_classes)]
with open(LABEL_PATH, "w", encoding="utf-8-sig") as f:
    for L in letters:
        f.write(L + "\n")
print("Wrote labels →", LABEL_PATH)

# ——— 4) ONE‑HOT + TRAIN/VAL SPLIT ———
y_cat = tf.keras.utils.to_categorical(y, n_classes)
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

# ——— 5) BUILD & TRAIN A TINY MLP ———
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(n_classes, activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32
)

# ——— 6) SAVE THE .keras MODEL ———
model.save(MODEL_PATH)
print("✅ Trained model saved to", MODEL_PATH)
