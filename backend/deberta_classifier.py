import pickle
import torch

from pathlib import Path
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "backend" / "saved_models" / "deberta_v3_model"

with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_DIR)

model = DebertaV2ForSequenceClassification.from_pretrained(
    MODEL_DIR
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model.to(device)
model.eval()


def classify_question(question):

    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[0][prediction].item()

    label = label_encoder.inverse_transform(
        [prediction]
    )[0]

    return {
        "label": label,
        "confidence": confidence
    }
