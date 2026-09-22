import pandas as pd
import matplotlib.pyplot as plt

models = [
    "LSTM V12",
    "DistilBERT V12",
    "DeBERTa-v2",
    "DeBERTa-v3 V12"
]

lstm = pd.read_csv("results/lstm_v12/test_predictions.csv")
distilbert = pd.read_csv("results/distilbert_v12/test_predictions.csv")
deberta_v2 = pd.read_csv("results/deberta_v2/test_predictions_v2.csv")

errors = [
    (~lstm["correct"]).sum(),
    (~distilbert["correct"]).sum(),
    (~deberta_v2["correct"]).sum(),
    53
]

plt.figure(figsize=(10, 6))

bars = plt.bar(models, errors)

plt.title("Test Error Comparison of FirstAidAI Models")
plt.xlabel("Model")
plt.ylabel("Number of Incorrect Predictions")

for bar, error in zip(bars, errors):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        str(error),
        ha="center"
    )

plt.tight_layout()

output = "results/error_comparison.png"
plt.savefig(output, dpi=300, bbox_inches="tight")
plt.close()

print("Error counts:")
for model, error in zip(models, errors):
    print(f"{model}: {error}")

print(f"Saved: {output}")
