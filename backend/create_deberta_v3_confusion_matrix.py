import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

classes = [
    "Bites & Stings",
    "Bleeding",
    "Burns",
    "CPR",
    "Choking",
    "Fractures & Injuries",
    "Neurological Emergencies",
    "Other",
    "Poisoning",
    "Shock & Fainting",
    "Temperature Emergencies"
]

cm = pd.read_csv(
    "results/deberta_v3/confusion_matrix.csv",
    header=None
).values

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

fig, ax = plt.subplots(figsize=(12, 10))
disp.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)

ax.set_title("DeBERTa-v3 Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

plt.tight_layout()

output = "results/deberta_v3/confusion_matrix_v3.png"
plt.savefig(output, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {output}")
