import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer


BASE_DIR = Path(__file__).resolve().parents[2]

df = pd.read_csv(
    BASE_DIR / "dataset" / "clean_firstaid_dataset_v6.csv"
)

X = df["question"].astype(str)
y = df["new_label"].astype(str)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
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


tokenizer = BertTokenizerFast.from_pretrained(
    "bert-base-uncased"
)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
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


model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(encoder.classes_)
)


def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(pred.label_ids, predictions)
    }


training_args = TrainingArguments(
    output_dir="../results/bert",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_dir="../results/logs_bert",
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


trainer.train()


predictions = trainer.predict(test_dataset)

y_pred = np.argmax(
    predictions.predictions,
    axis=1
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("BERT TEST RESULTS")
print("=" * 60)

print("\nTest Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

fig, ax = plt.subplots(figsize=(12, 10))

disp.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title("BERT Confusion Matrix")
plt.tight_layout()

plt.savefig(
    BASE_DIR / "backend" / "results" / "bert_confusion_matrix.png"
)

plt.close()


model_path = (
    BASE_DIR
    / "backend"
    / "saved_models"
    / "bert_model"
)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

with open(
    BASE_DIR
    / "backend"
    / "saved_models"
    / "bert_label_encoder.pkl",
    "wb"
) as f:
    pickle.dump(encoder, f)

print("\nBERT model saved successfully.")
