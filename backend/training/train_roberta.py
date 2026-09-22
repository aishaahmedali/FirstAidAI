import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import Dataset

from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer
)

import torch

BASE_DIR = Path(__file__).resolve().parents[2]

print(BASE_DIR)

df = pd.read_csv(BASE_DIR / "dataset" / "clean_firstaid_dataset.csv")

X = df["question"].astype(str)
y = df["new_label"]

print(df["new_label"].value_counts())
print("Classes:", df["new_label"].nunique())

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

train_dataset = Dataset.from_dict({
    "text": X_train.tolist(),
    "label": y_train.tolist()
})

val_dataset = Dataset.from_dict({
    "text": X_val.tolist(),
    "label": y_val.tolist()
})

test_dataset = Dataset.from_dict({
    "text": X_test.tolist(),
    "label": y_test.tolist()
})

tokenizer = RobertaTokenizerFast.from_pretrained(
    "roberta-base"
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)

val_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)

test_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=len(encoder.classes_)
)

training_args = TrainingArguments(
    output_dir="../results/roberta",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=1,
    logging_dir="../results/roberta_logs"
)

def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids

    return {
        "accuracy": accuracy_score(labels, predictions)
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

predictions = trainer.predict(test_dataset)

y_pred = np.argmax(predictions.predictions, axis=1)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)

model.save_pretrained("../saved_models/roberta_model")

tokenizer.save_pretrained("../saved_models/roberta_model")

with open("../saved_models/roberta_label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("RoBERTa model saved successfully.")

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(ax=ax, xticks_rotation=90)

plt.title("RoBERTa Confusion Matrix")
plt.tight_layout()
plt.savefig("../results/roberta_confusion_matrix.png")
plt.show()