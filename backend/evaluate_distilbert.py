import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from datasets import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer
)

df = pd.read_csv("../../dataset/labeled_firstaid_dataset.csv")

small_classes = [
    "Electric Shock",
    "Allergic Reaction"
]

df["label"] = df["label"].replace(
    small_classes,
    "Other"
)

X = df["question"].astype(str)
y = df["label"]

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

test_dataset = Dataset.from_dict({
    "text": X_test.tolist(),
    "label": y_test.tolist()
})

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "../saved_models/distilbert_model",
    local_files_only=True
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

test_dataset = test_dataset.map(tokenize, batched=True)

test_dataset.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)


model = DistilBertForSequenceClassification.from_pretrained(
    "../saved_models/distilbert_model",
    local_files_only=True
)
trainer = Trainer(model=model)

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

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

fig, ax = plt.subplots(figsize=(12, 12))

disp.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues"
)

plt.title("DistilBERT Confusion Matrix")
plt.tight_layout()

plt.savefig("results/distilbert_confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved successfully.")