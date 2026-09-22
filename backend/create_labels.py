import pandas as pd

df = pd.read_csv("../dataset/final_firstaid_dataset.csv")

categories = {
    "Spinal Injury": [
        "spinal", "spine", "neck injury", "back injury",
        "neck", "backboard", "casualty movement", "move casualty"
    ],

    "Bleeding": [
        "bleeding", "bleed", "blood", "hemorrhage",
        "wound", "cut", "laceration", "tourniquet",
        "direct pressure"
    ],

    "Burns": [
        "burn", "burns", "scald", "hot water",
        "chemical burn", "electrical burn", "sunburn"
    ],

    "CPR": [
        "cpr", "cardiac arrest", "rescue breaths",
        "chest compression", "compressions",
        "mouth-to-mouth"
    ],

    "Choking": [
        "choking", "heimlich", "cannot breathe",
        "airway obstruction", "abdominal thrust"
    ],

    "Fracture": [
        "fracture", "broken", "bone", "sprain",
        "dislocation", "splint", "cast"
    ],

    "Poisoning": [
        "poison", "poisoning", "toxic",
        "chemical", "overdose"
    ],

    "Snake Bite": [
        "snake", "venom", "snake bite"
    ],

    "Insect Bite": [
        "bee", "wasp", "sting", "spider",
        "mosquito", "tick", "insect"
    ],

    "Heat Stroke": [
        "heat stroke", "heat exhaustion",
        "overheating"
    ],

    "Hypothermia": [
        "hypothermia", "cold exposure",
        "frostbite"
    ],

    "Shock": [
        "shock"
    ],

    "Seizure": [
        "seizure", "convulsion", "epilepsy"
    ],

    "Fainting": [
        "faint", "fainting", "passed out",
        "syncope"
    ],

    "Eye Injury": [
        "eye", "vision", "chemical in eye"
    ],

    "Dental Injury": [
        "tooth", "teeth", "dental"
    ],

    "Electric Shock": [
        "electric shock", "electrocution",
        "electricity"
    ],

    "Drowning": [
        "drowning", "water rescue"
    ],

    "Animal Bite": [
        "dog bite", "cat bite",
        "animal bite", "rabies"
    ],

    "Allergic Reaction": [
        "allergic", "allergy",
        "anaphylaxis"
    ]
}


def assign_label(question, answer):

    text = (str(question) + " " + str(answer)).lower()

    for label, keywords in categories.items():
        for keyword in keywords:
            if keyword in text:
                return label

    return "Other"


df["label"] = df.apply(
    lambda row: assign_label(row["question"], row["answer"]),
    axis=1
)

print(df["label"].value_counts())

df.to_csv(
    "../dataset/labeled_firstaid_dataset.csv",
    index=False
)

print("Dataset saved successfully!")