import pickle
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "dataset" / "final_split" / "test.csv"
MODEL_DIR = BASE_DIR / "backend" / "saved_models" / "deberta_v3_model"

df = pd.read_csv(TEST_FILE)

with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_DIR)
model = DebertaV2ForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

questions = df["question"].astype(str).tolist()

predictions = []

batch_size = 8

for i in range(0, len(questions), batch_size):
    batch = questions[i:i + batch_size]

    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    predictions.extend(preds)

df["predicted_label"] = label_encoder.inverse_transform(predictions)

errors = df[df["new_label"] != df["predicted_label"]].copy()

print("=" * 80)
print("DEBERTA ERROR ANALYSIS")
print("=" * 80)

print("\nTotal test samples:", len(df))
print("Correct:", len(df) - len(errors))
print("Incorrect:", len(errors))

print("\nErrors by TRUE -> PREDICTED:")
print(
    errors.groupby(
        ["new_label", "predicted_label"]
    ).size().sort_values(ascending=False)
)

print("\n" + "=" * 80)
print("ALL INCORRECT PREDICTIONS")
print("=" * 80)

for i, row in errors.iterrows():
    print("\n" + "-" * 80)
    print("TRUE      :", row["new_label"])
    print("PREDICTED :", row["predicted_label"])
    print("QUESTION  :", row["question"])

errors.to_csv(
    BASE_DIR / "results" / "deberta_v1" / "misclassified_test_samples.csv",
    index=False
)

print("\nSaved:")
print(
    BASE_DIR
    / "results"
    / "deberta_v1"
    / "misclassified_test_samples.csv"
)
