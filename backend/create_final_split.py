import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
input_file = BASE_DIR / "dataset" / "clean_firstaid_dataset_v12.csv"
split_dir = BASE_DIR / "dataset" / "final_split_v12"

split_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)
df["question"] = df["question"].astype(str).str.strip()
df["new_label"] = df["new_label"].astype(str).str.strip()

print("=" * 60)
print("CREATING FINAL LEAKAGE-SAFE SPLIT")
print("=" * 60)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1
)

X = vectorizer.fit_transform(df["question"])

similarity = cosine_similarity(X)

parent = list(range(len(df)))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra = find(a)
    rb = find(b)
    if ra != rb:
        parent[rb] = ra

threshold = 0.90

for i in range(len(df)):
    for j in range(i + 1, len(df)):
        if similarity[i, j] >= threshold:
            union(i, j)

groups = {}

for i in range(len(df)):
    root = find(i)
    groups.setdefault(root, []).append(i)

group_ids = list(groups.keys())

print("Total samples:", len(df))
print("Similarity groups:", len(group_ids))
print("Grouped samples:", sum(len(v) > 1 for v in groups.values()))

group_table = pd.DataFrame({
    "group_id": list(range(len(group_ids))),
    "root": group_ids
})

root_to_group = {
    root: idx for idx, root in enumerate(group_ids)
}

df["group_id"] = [root_to_group[find(i)] for i in range(len(df))]

group_labels = (
    df.groupby("group_id")["new_label"]
    .first()
    .reset_index()
)

train_groups, temp_groups = train_test_split(
    group_labels,
    test_size=0.20,
    random_state=42,
    stratify=group_labels["new_label"]
)

val_groups, test_groups = train_test_split(
    temp_groups,
    test_size=0.50,
    random_state=42,
    stratify=temp_groups["new_label"]
)

train_ids = set(train_groups["group_id"])
val_ids = set(val_groups["group_id"])
test_ids = set(test_groups["group_id"])

train_df = df[df["group_id"].isin(train_ids)].copy()
val_df = df[df["group_id"].isin(val_ids)].copy()
test_df = df[df["group_id"].isin(test_ids)].copy()

train_df = train_df.drop(columns=["group_id"])
val_df = val_df.drop(columns=["group_id"])
test_df = test_df.drop(columns=["group_id"])

train_df.to_csv(split_dir / "train.csv", index=False)
val_df.to_csv(split_dir / "validation.csv", index=False)
test_df.to_csv(split_dir / "test.csv", index=False)

print()
print("TRAIN:", len(train_df))
print("VALIDATION:", len(val_df))
print("TEST:", len(test_df))

print()
print("TRAIN DISTRIBUTION:")
print(train_df["new_label"].value_counts())

print()
print("VALIDATION DISTRIBUTION:")
print(val_df["new_label"].value_counts())

print()
print("TEST DISTRIBUTION:")
print(test_df["new_label"].value_counts())

print()
print("Saved:")
print(split_dir / "train.csv")
print(split_dir / "validation.csv")
print(split_dir / "test.csv")

print("=" * 60)
print("FINAL SPLIT CREATED")
print("=" * 60)
