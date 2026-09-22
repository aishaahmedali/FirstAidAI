import pickle
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from datasets import Dataset
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_DIR = BASE_DIR / "backend" / "saved_models" / "deberta_v2_model"
TEST_FILE = BASE_DIR / "dataset" / "final_split_v10" / "test.csv"
RESULTS_DIR = BASE_DIR / "backend" / "results" / "deberta_v2"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DeBERTa-v2 MODEL EVALUATION")
print("=" * 70)

test_df = pd.read_csv(TEST_FILE)

with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

test_df["label"] = label_encoder.transform(test_df["new_label"])

test_dataset = Dataset.from_pandas(
    test_df[["question", "label"]],
    preserve_index=False
)

tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_DIR)

def tokenize(batch):
    return tokenizer(
        batch["question"],
        truncation=True,
        max_length=128
    )

test_dataset = test_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["question"]
)

model = DebertaV2ForSequenceClassification.from_pretrained(
    MODEL_DIR
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("\nEvaluating TEST set...")

result = trainer.predict(test_dataset)

predictions = np.argmax(
    result.predictions,
    axis=1
)

labels = result.label_ids

accuracy = accuracy_score(labels, predictions)

print("\nTest Accuracy:", accuracy)

report = classification_report(
    labels,
    predictions,
    target_names=label_encoder.classes_,
    digits=4
)

print("\nClassification Report:")
print(report)

cm = confusion_matrix(labels, predictions)

np.savetxt(
    RESULTS_DIR / "confusion_matrix_v2.csv",
    cm,
    delimiter=",",
    fmt="%d"
)

with open(
    RESULTS_DIR / "classification_report_v2.txt",
    "w"
) as f:
    f.write(report)

results_df = test_df[["question", "new_label"]].copy()

results_df["predicted_label"] = label_encoder.inverse_transform(
    predictions
)

results_df["correct"] = (
    results_df["new_label"] ==
    results_df["predicted_label"]
)

results_df.to_csv(
    RESULTS_DIR / "test_predictions_v2.csv",
    index=False
)

errors = results_df[
    results_df["correct"] == False
]

errors.to_csv(
    RESULTS_DIR / "misclassified_test_samples_v2.csv",
    index=False
)

print("\nTotal test samples:", len(results_df))
print("Correct:", results_df["correct"].sum())
print("Incorrect:", (~results_df["correct"]).sum())

print("\nSaved:")
print(RESULTS_DIR / "confusion_matrix_v2.csv")
print(RESULTS_DIR / "classification_report_v2.txt")
print(RESULTS_DIR / "test_predictions_v2.csv")
print(RESULTS_DIR / "misclassified_test_samples_v2.csv")

print("\nEvaluation complete.")
