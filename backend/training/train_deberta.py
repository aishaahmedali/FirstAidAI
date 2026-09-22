import os
import random
import pickle
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datasets import Dataset
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

TRAIN_FILE = BASE_DIR / "dataset" / "final_split_v12" / "train.csv"
VAL_FILE = BASE_DIR / "dataset" / "final_split_v12" / "validation.csv"
TEST_FILE = BASE_DIR / "dataset" / "final_split_v12" / "test.csv"

MODEL_NAME = "microsoft/deberta-v3-base"

MODEL_DIR = BASE_DIR / "backend" / "saved_models" / "deberta_v3_v10_model"
RESULTS_DIR = BASE_DIR / "backend" / "results" / "deberta_v3_c10"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("DeBERTa-v3-base V10 TRAINING")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

train_df = train_df[["question", "new_label"]].copy()
val_df = val_df[["question", "new_label"]].copy()
test_df = test_df[["question", "new_label"]].copy()

train_df["question"] = train_df["question"].astype(str).str.strip()
val_df["question"] = val_df["question"].astype(str).str.strip()
test_df["question"] = test_df["question"].astype(str).str.strip()

label_encoder = LabelEncoder()

train_df["label"] = label_encoder.fit_transform(train_df["new_label"])

label_map = dict(
    zip(
        label_encoder.classes_,
        range(len(label_encoder.classes_))
    )
)

val_df["label"] = val_df["new_label"].map(label_map)
test_df["label"] = test_df["new_label"].map(label_map)

if val_df["label"].isna().any():
    raise ValueError("Validation set contains unknown labels")

if test_df["label"].isna().any():
    raise ValueError("Test set contains unknown labels")

val_df["label"] = val_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)

print("\nClasses:")
for i, label in enumerate(label_encoder.classes_):
    print(i, "->", label)

print("\nDataset sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTrain distribution:")
print(train_df["new_label"].value_counts())

train_dataset = Dataset.from_pandas(
    train_df[["question", "label"]],
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df[["question", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["question", "label"]],
    preserve_index=False
)

print("\nLoading tokenizer...")

tokenizer = DebertaV2Tokenizer.from_pretrained(
    MODEL_NAME
)

def tokenize(batch):
    return tokenizer(
        batch["question"],
        truncation=True,
        max_length=160
    )

train_dataset = train_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["question"]
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["question"]
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["question"]
)

num_labels = len(label_encoder.classes_)

id2label = {
    i: label
    for i, label in enumerate(label_encoder.classes_)
}

label2id = {
    label: i
    for i, label in enumerate(label_encoder.classes_)
}

print("\nLoading DeBERTa model...")

model = DebertaV2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    use_safetensors=True
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

def compute_metrics(pred):
    predictions = np.argmax(
        pred.predictions,
        axis=1
    )

    labels = pred.label_ids

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        )
    }

training_args = TrainingArguments(
    output_dir=str(RESULTS_DIR),

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=1e-5,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,

    gradient_accumulation_steps=2,

    num_train_epochs=10,

    weight_decay=0.01,

    warmup_ratio=0.1,

    lr_scheduler_type="cosine",

    label_smoothing_factor=0.05,

    logging_strategy="steps",
    logging_steps=50,

    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    save_total_limit=2,

    report_to="none",

    fp16=False,

    seed=SEED,

    dataloader_num_workers=0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
)

print("\nStarting training...")
print("=" * 70)

trainer.train()

print("\nBest model:")
print(trainer.state.best_model_checkpoint)

print("\nEvaluating validation set...")

val_predictions = trainer.predict(val_dataset)

val_pred = np.argmax(
    val_predictions.predictions,
    axis=1
)

val_true = val_predictions.label_ids

val_accuracy = accuracy_score(
    val_true,
    val_pred
)

print("\nValidation Accuracy:", val_accuracy)

print("\nEvaluating TEST set...")

test_predictions = trainer.predict(test_dataset)

test_pred = np.argmax(
    test_predictions.predictions,
    axis=1
)

test_true = test_predictions.label_ids

test_accuracy = accuracy_score(
    test_true,
    test_pred
)

print("\nTest Accuracy:", test_accuracy)

print("\nClassification Report:")

report = classification_report(
    test_true,
    test_pred,
    target_names=label_encoder.classes_,
    digits=4
)

print(report)

cm = confusion_matrix(
    test_true,
    test_pred
)

np.savetxt(
    RESULTS_DIR / "confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)

with open(
    RESULTS_DIR / "classification_report.txt",
    "w"
) as f:
    f.write(report)

with open(
    MODEL_DIR / "label_encoder.pkl",
    "wb"
) as f:
    pickle.dump(label_encoder, f)

trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

print("\nConfusion matrix saved.")

print("\nModel saved to:")
print(MODEL_DIR)

print("\nTraining complete.")