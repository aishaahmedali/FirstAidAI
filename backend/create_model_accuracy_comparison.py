import matplotlib.pyplot as plt

models = [
    "LSTM V12",
    "DistilBERT V12",
    "DeBERTa-v2",
    "DeBERTa-v3 V12"
]

accuracies = [
    58.00,
    89.00,
    89.22,
    91.82
]

plt.figure(figsize=(10, 6))

bars = plt.bar(models, accuracies)

plt.title("Test Accuracy Comparison of FirstAidAI Models")
plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 100)

for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.tight_layout()

output = "results/model_accuracy_comparison.png"
plt.savefig(output, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {output}")
