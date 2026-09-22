import pandas as pd
import numpy as np
import pickle
import torch

from pathlib import Path
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "dataset" / "final_split_v12" / "test.csv"
MODEL_DIR = BASE_DIR / "backend" / "saved_models" / "distilbert_v7_model"
LABEL_ENCODER = BASE_DIR / "backend" / "saved_models" / "distilbert_v7_label_encoder.pkl"
RESULTS_DIR = BASE_DIR / "backend" / "results" / "distilbert_v12"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TEST_FILE)

with open(LABEL_ENCODER, "rb") as f:
    label_encoder = pickle.load(f)

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model.to(device)
model.eval()

questions = df["question"].astype(str).tolist()

predictions = []

batch_size = 16

for i in range(0, len(questions), batch_size):

    batch = questions[i:i + batch_size]

    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    preds = torch.argmax(
        outputs.logits,
        dim=1
    ).cpu().numpy()

    predictions.extend(preds)

y_true = df["new_label"].values
y_pred = label_encoder.inverse_transform(predictions)

accuracy = accuracy_score(y_true, y_pred)

print("=" * 80)
print("DISTILBERT V7 — V12 TEST SET")
print("=" * 80)

print("\nTest samples:", len(df))
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=label_encoder.classes_,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=label_encoder.classes_
)

pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
).to_csv(
    RESULTS_DIR / "confusion_matrix.csv"
)

results = df.copy()
results["predicted_label"] = y_pred
results["correct"] = results["new_label"] == results["predicted_label"]

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
            labels=label_encoder.classes_,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

print("\nResults saved to:")
print(RESULTS_DIR)
