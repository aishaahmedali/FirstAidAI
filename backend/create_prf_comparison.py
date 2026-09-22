import matplotlib.pyplot as plt
import numpy as np

models = [
    "LSTM V12",
    "DistilBERT V12",
    "DeBERTa-v2",
    "DeBERTa-v3 V12"
]

precision = [51.00, 89.00, 89.59, 92.27]
recall = [49.00, 91.00, 88.31, 91.15]
f1 = [42.00, 90.00, 88.86, 91.67]

x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(11, 6))

plt.bar(x - width, precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar(x + width, f1, width, label="F1-Score")

plt.title("Precision, Recall and F1-Score Comparison")
plt.xlabel("Model")
plt.ylabel("Score (%)")
plt.xticks(x, models)
plt.ylim(0, 100)
plt.legend()

plt.tight_layout()

output = "results/precision_recall_f1_comparison.png"
plt.savefig(output, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {output}")
