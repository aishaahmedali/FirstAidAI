from pathlib import Path
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "dataset" / "clean_firstaid_dataset.csv"
MODEL_DIR = BASE_DIR / "backend" / "saved_models"

df = pd.read_csv(DATASET_PATH)

print(df["new_label"].value_counts())
print("Classes:", df["new_label"].nunique())

X = df["question"].astype(str)
y = df["new_label"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 40

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train_pad,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[early_stop]
)

predictions = model.predict(X_test_pad)
y_pred = np.argmax(predictions, axis=1)

accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy)
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
print(confusion_matrix(y_test, y_pred))

MODEL_DIR.mkdir(parents=True, exist_ok=True)

model.save(MODEL_DIR / "lstm_firstaid_model.keras")

with open(MODEL_DIR / "lstm_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open(MODEL_DIR / "lstm_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Model saved successfully!")