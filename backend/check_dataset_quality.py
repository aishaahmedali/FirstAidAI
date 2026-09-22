import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
df = pd.read_csv(BASE_DIR / "dataset" / "clean_firstaid_dataset_v7.csv")
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print("Total samples:", len(df))
print("Total classes:", df["new_label"].nunique())

print("\nClass distribution:")
print(df["new_label"].value_counts())

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("DUPLICATES")
print("=" * 50)

dup_questions = df["question"].duplicated().sum()
dup_pairs = df.duplicated(subset=["question", "answer"]).sum()

print("Duplicate questions:", dup_questions)
print("Duplicate question-answer pairs:", dup_pairs)

print("\n" + "=" * 50)
print("QUESTION LENGTHS")
print("=" * 50)

df["question_len"] = df["question"].astype(str).apply(lambda x: len(x.split()))

print(df["question_len"].describe())

print("\nVery short questions (<5 words):")
short_q = df[df["question_len"] < 5]
print(short_q[["question", "new_label"]].head(20))

print("\nVery long questions (>25 words):")
long_q = df[df["question_len"] > 25]
print(long_q[["question", "new_label"]].head(20))

print("\n" + "=" * 50)
print("POSSIBLE LABEL ISSUES (SAME QUESTION, DIFFERENT LABEL)")
print("=" * 50)

label_conflicts = (
    df.groupby("question")["new_label"]
    .nunique()
    .reset_index()
)

label_conflicts = label_conflicts[label_conflicts["new_label"] > 1]

print("Conflicting questions:", len(label_conflicts))

if len(label_conflicts) > 0:
    for q in label_conflicts["question"].head(10):
        print("\nQUESTION:", q)
        print(df[df["question"] == q][["question", "new_label"]])

print("\n" + "=" * 50)
print("SIMILAR QUESTIONS WITH DIFFERENT LABELS")
print("=" * 50)

sample_df = df.sample(min(1000, len(df)), random_state=42).reset_index(drop=True)

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(sample_df["question"].astype(str))

sim = cosine_similarity(X)

suspicious = []

for i in range(len(sample_df)):
    for j in range(i + 1, len(sample_df)):
        if sim[i, j] > 0.75:
            if sample_df.loc[i, "new_label"] != sample_df.loc[j, "new_label"]:
                suspicious.append({
                    "q1": sample_df.loc[i, "question"],
                    "label1": sample_df.loc[i, "new_label"],
                    "q2": sample_df.loc[j, "question"],
                    "label2": sample_df.loc[j, "new_label"],
                    "similarity": round(sim[i, j], 3)
                })

print(f"Found {len(suspicious)} suspicious similar pairs")

for item in suspicious[:20]:
    print("\nSIMILARITY:", item["similarity"])
    print("Q1:", item["q1"])
    print("L1:", item["label1"])
    print("Q2:", item["q2"])
    print("L2:", item["label2"])

print("\n" + "=" * 50)
print("OTHER CLASS INSPECTION")
print("=" * 50)

other_df = df[df["new_label"] == "Other"]

print("Other samples:", len(other_df))

print("\nRandom samples from Other:")
print(other_df[["question"]].sample(min(20, len(other_df)), random_state=42).to_string(index=False))

print("\nDone.")