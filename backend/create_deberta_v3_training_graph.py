import json
import matplotlib.pyplot as plt

path = "backend/results/deberta_v3/checkpoint-5152/trainer_state.json"

with open(path, "r") as f:
    data = json.load(f)

history = data["log_history"]

train_epochs = []
train_loss = []
eval_epochs = []
eval_loss = []
eval_accuracy = []

for entry in history:

    if "loss" in entry and "epoch" in entry:
        train_epochs.append(entry["epoch"])
        train_loss.append(entry["loss"])

    if "eval_accuracy" in entry:
        eval_epochs.append(entry["epoch"])
        eval_accuracy.append(entry["eval_accuracy"] * 100)
        eval_loss.append(entry["eval_loss"])

fig, ax1 = plt.subplots(figsize=(11, 6))

line1 = ax1.plot(
    train_epochs,
    train_loss,
    label="Training Loss"
)

line2 = ax1.plot(
    eval_epochs,
    eval_loss,
    marker="o",
    label="Validation Loss"
)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("DeBERTa-v3 Training and Validation Performance")

ax2 = ax1.twinx()

line3 = ax2.plot(
    eval_epochs,
    eval_accuracy,
    marker="o",
    label="Validation Accuracy"
)

ax2.set_ylabel("Validation Accuracy (%)")
ax2.set_ylim(0, 100)

lines = line1 + line2 + line3
labels = [line.get_label() for line in lines]

ax1.legend(
    lines,
    labels,
    loc="center right"
)

plt.tight_layout()

output = "backend/results/deberta_v3/training_validation_graph.png"

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Training loss points: {len(train_loss)}")
print(f"Validation points: {len(eval_epochs)}")
print(f"Best validation accuracy: {max(eval_accuracy):.2f}%")
print(f"Saved: {output}")