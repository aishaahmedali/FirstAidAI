from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "dataset" / "final_split_v12" / "test.csv"

MODEL_FILE = BASE_DIR / "backend" / "saved_models" / "lstm_firstaid_model.keras"
TOKENIZER_FILE = BASE_DIR / "backend" / "saved_models" / "lstm_tokenizer.pkl"
ENCODER_FILE = BASE_DIR / "backend" / "saved_models" / "lstm_encoder.pkl"

RESULTS_DIR = BASE_DIR / "backend" / "results" / "lstm_v12"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TEST_FILE)

model = load_model(MODEL_FILE)

with open(TOKENIZER_FILE, "rb") as f:
    tokenizer = pickle.load(f)

with open(ENCODER_FILE, "rb") as f:
    encoder = pickle.load(f)

questions = df["question"].astype(str)

X_seq = tokenizer.texts_to_sequences(questions)

X_pad = pad_sequences(
    X_seq,
    maxlen=40,
    padding="post",
    truncating="post"
)

predictions = model.predict(
    X_pad,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)

y_true = encoder.transform(df["new_label"])

accuracy = accuracy_score(y_true, y_pred)

print("=" * 80)
print("LSTM — V12 TEST SET")
print("=" * 80)

print("\nTest samples:", len(df))
print("Accuracy:", accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(encoder.classes_))
)

pd.DataFrame(
    cm,
    index=encoder.classes_,
    columns=encoder.classes_
).to_csv(
    RESULTS_DIR / "confusion_matrix.csv"
)

results = df.copy()

results["predicted_label"] = encoder.inverse_transform(y_pred)
results["correct"] = (
    results["new_label"] == results["predicted_label"]
)

results.to_csv(
    RESULTS_DIR / "test_predictions.csv",
    index=False
)

with open(
    RESULTS_DIR / "classification_report.txt",
    "w"
) as f:
    f.write(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0
        )
    )

print("\nResults saved to:")
print(RESULTS_DIR)
