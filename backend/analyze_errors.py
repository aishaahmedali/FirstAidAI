import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer

BASE_DIR = Path(__file__).resolve().parents[1]

df = pd.read_csv(BASE_DIR / "dataset" / "clean_firstaid_dataset_v7.csv")

X = df["question"].astype(str)
y = df["new_label"]

with open(BASE_DIR / "backend" / "saved_models" / "distilbert_v3_label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

y_encoded = encoder.transform(y)

X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
    X,
    y_encoded,
    df.index,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
    X_temp,
    y_temp,
    idx_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

test_dataset = Dataset.from_dict({
    "text": X_test.tolist(),
    "label": y_test.tolist()
})

tokenizer = DistilBertTokenizerFast.from_pretrained(
    BASE_DIR / "backend" / "saved_models" / "distilbert_v3_model"
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
    BASE_DIR / "backend" / "saved_models" / "distilbert_v3_model"
)

trainer = Trainer(model=model)

predictions = trainer.predict(test_dataset)

y_pred = np.argmax(predictions.predictions, axis=1)

results = pd.DataFrame({
    "question": X_test.values,
    "true_label": encoder.inverse_transform(y_test),
    "predicted_label": encoder.inverse_transform(y_pred)
})

errors = results[results["true_label"] != results["predicted_label"]]

print("\nTotal errors:", len(errors))
print("\n==============================")
print("BLEEDING → FRACTURES & INJURIES")
print("==============================")

print(
    errors[
        (errors["true_label"] == "Bleeding") &
        (errors["predicted_label"] == "Fractures & Injuries")
    ][["question", "true_label", "predicted_label"]].to_string(index=False)
)

print("\n==============================")
print("BLEEDING → OTHER")
print("==============================")

print(
    errors[
        (errors["true_label"] == "Bleeding") &
        (errors["predicted_label"] == "Other")
    ][["question", "true_label", "predicted_label"]].to_string(index=False)
)

print("\n==============================")
print("FRACTURES & INJURIES → BLEEDING")
print("==============================")

print(
    errors[
        (errors["true_label"] == "Fractures & Injuries") &
        (errors["predicted_label"] == "Bleeding")
    ][["question", "true_label", "predicted_label"]].to_string(index=False)
)

print("\n==============================")
print("FRACTURES & INJURIES → OTHER")
print("==============================")

print(
    errors[
        (errors["true_label"] == "Fractures & Injuries") &
        (errors["predicted_label"] == "Other")
    ][["question", "true_label", "predicted_label"]].to_string(index=False)
)

print("\n==============================")
print("BITES & STINGS → OTHER")
print("==============================")

print(
    errors[
        (errors["true_label"] == "Bites & Stings") &
        (errors["predicted_label"] == "Other")
    ][["question", "true_label", "predicted_label"]].to_string(index=False)
)