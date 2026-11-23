from pathlib import Path
import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification

model_path = Path("backend/saved_models/deberta_v3_model")

tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
model = DebertaV2ForSequenceClassification.from_pretrained(model_path)
model.eval()

question = "What should I do if someone has a severe burn?"

inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.softmax(outputs.logits, dim=1)
prediction = probabilities.argmax(dim=1).item()

print("\nFIRSTAIDAI CLASSIFIER TEST")
print("--------------------------")
print("Question:", question)
print("Predicted category:", model.config.id2label[prediction])
print("Confidence:", round(probabilities[0][prediction].item() * 100, 2), "%")
