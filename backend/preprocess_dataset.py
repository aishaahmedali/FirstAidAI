import json
import pandas as pd


with open("../dataset/training/firstaidqa_v1.json", "r") as f:
    json_data = json.load(f)

json_df = pd.DataFrame(json_data)


csv_df = pd.read_csv("../dataset/training/merged_data.csv")


if "Unnamed: 0" in csv_df.columns:
    csv_df = csv_df.drop(columns=["Unnamed: 0"])


json_df = json_df.dropna()
csv_df = csv_df.dropna()


json_df = json_df[["question", "answer"]]
csv_df = csv_df[["question", "answer"]]


final_df = pd.concat(
    [json_df, csv_df],
    ignore_index=True
)


final_df = final_df.drop_duplicates(
    subset=["question"]
)


final_df.to_csv(
    "../dataset/final_firstaid_dataset.csv",
    index=False
)


print("Dataset created successfully!")
print("Final size:", len(final_df))