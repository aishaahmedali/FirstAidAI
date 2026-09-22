import pandas as pd
import numpy as np
import pickle
import torch

from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]

df = pd.read_csv(BASE_DIR / "dataset" / "clean_firstaid_dataset_v6.csv")

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

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
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

class_counts = np.bincount(y_train)
weights = len(y_train) / (len(class_counts) * class_counts)
class_weights = torch.tensor(weights, dtype=torch.float)

print("Class weights:")
for label, weight in zip(encoder.classes_, class_weights):
    print(f"{label}: {weight:.4f}")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(encoder.classes_)
)

class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device)
        )

        loss = loss_function(
            logits.view(-1, model.config.num_labels),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="../results/distilbert",
    logging_dir="../results/logs",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none"
)

def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids

    return {
        "accuracy": accuracy_score(labels, predictions)
    }

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

predictions = trainer.predict(test_dataset)

y_pred = np.argmax(predictions.predictions, axis=1)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)

model_path = BASE_DIR / "backend" / "saved_models" / "distilbert_v7_model"

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

with open(
    BASE_DIR / "backend" / "saved_models" / "distilbert_v7_label_encoder.pkl",
    "wb"
) as f:
    pickle.dump(encoder, f)

print("DistilBERT v7 model saved successfully.")

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

fig, ax = plt.subplots(figsize=(12, 12))

disp.plot(
    ax=ax,
    xticks_rotation=90
)

plt.title("DistilBERT v7 Confusion Matrix")
plt.tight_layout()

plt.savefig(
    BASE_DIR / "backend" / "results" / "distilbert_v7_confusion_matrix.png"
)

plt.show()
