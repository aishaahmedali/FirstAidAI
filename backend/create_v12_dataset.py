from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT = BASE_DIR / "dataset" / "clean_firstaid_dataset_v11.csv"
OUTPUT = BASE_DIR / "dataset" / "clean_firstaid_dataset_v12.csv"

df = pd.read_csv(INPUT)

diabetes_keywords = [
    "diabet",
    "blood sugar",
    "hypogly",
    "hypergly",
    "glucose",
    "insulin",
    "diabetic ketoacidosis",
    "ketoacidosis",
    "low sugar",
    "high sugar",
    "fruity odor",
    "fruity breath",
    "sweet-smelling breath"
]

mask = (
    df["new_label"].eq("Bleeding")
    & df["question"].str.contains(
        "|".join(diabetes_keywords),
        case=False,
        na=False
    )
)

print("=" * 70)
print("CREATING V11 DATASET")
print("=" * 70)

print("Bleeding samples before:", (df["new_label"] == "Bleeding").sum())
print("Diabetes samples moved:", mask.sum())

df.loc[mask, "new_label"] = "Other"

choking_keywords = [
    "swallow",
    "swallowed object",
    "swallowed a",
    "swallowing",
    "button battery",
    "sharp object",
    "small metal",
    "piece of metal",
    "piece of glass",
    "maintain.*airway",
    "keep.*airway",
    "open.*airway",
    "obstructed airway",
    "airway.*obstruct",
    "manage.*airway",
    "checking.*airway",
    "check.*airway"
]

choking_mask = (
    df["new_label"].eq("Bleeding")
    & df["question"].str.contains(
        "|".join(choking_keywords),
        case=False,
        na=False,
        regex=True
    )
)

print("Choking samples moved:", choking_mask.sum())

df.loc[choking_mask, "new_label"] = "Choking"

df = df.drop_duplicates(
    subset=["question", "new_label"]
).reset_index(drop=True)

df.to_csv(OUTPUT, index=False)

print()
print("Saved:", OUTPUT)
print("Total samples:", len(df))
print("Total classes:", df["new_label"].nunique())
print()
print(df["new_label"].value_counts())
print()
print("Duplicate questions:", df["question"].duplicated().sum())
print("Duplicate pairs:", df.duplicated(["question", "new_label"]).sum())
