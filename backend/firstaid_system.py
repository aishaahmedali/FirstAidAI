import os
import pickle
from pathlib import Path

import chromadb
import torch
from google import genai
from sentence_transformers import SentenceTransformer
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "backend" / "saved_models" / "deberta_v3_model"
VECTOR_DB = BASE_DIR / "vector_db"

GEMINI_MODEL = "models/gemini-3.5-flash"
COLLECTION_NAME = "first_aid"


with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_DIR)

model = DebertaV2ForSequenceClassification.from_pretrained(
    MODEL_DIR
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model.to(device)
model.eval()


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


chroma_client = chromadb.PersistentClient(
    path=str(VECTOR_DB)
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )

gemini = genai.Client(
    api_key=api_key
)


def classify_question(question):

    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[0][prediction].item()

    label = label_encoder.inverse_transform(
        [prediction]
    )[0]

    return label, confidence


def retrieve_context(question, category):

    search_query = f"{category} first aid treatment: {question}"

    question_embedding = embedding_model.encode(
        search_query
    )

    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=3
    )

    documents = results.get("documents", [[]])[0]

    if not documents:
        return None

    return "\n\n".join(documents)


def generate_answer(question, category, context):

    prompt = f"""
You are FirstAidAI, an AI assistant for first-aid guidance.

The question has been classified as:

Category: {category}

Use the retrieved first-aid information below to answer the user's question.

Retrieved first-aid information:
{context}

User question:
{question}

Instructions:
- Give clear, practical first-aid guidance.
- Base the answer on the retrieved information.
- Do not invent medical instructions that are not supported by the retrieved information.
- Use numbered steps when appropriate.
- If the situation appears life-threatening, clearly advise the user to seek emergency medical help.
- Do not claim to diagnose the user.

Answer:
"""

    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


def ask_firstaid(question):

    category, confidence = classify_question(
        question
    )

    context = retrieve_context(
        question,
        category
    )

    if context is None:
        return {
            "question": question,
            "category": category,
            "confidence": confidence,
            "answer": "I couldn't find relevant information in the First Aid Guide."
        }

    answer = generate_answer(
        question,
        category,
        context
    )

    return {
        "question": question,
        "category": category,
        "confidence": confidence,
        "context": context,
        "answer": answer
    }


def main():

    print("=" * 60)
    print("FIRSTAIDAI")
    print("=" * 60)

    question = input("\nAsk a first aid question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    result = ask_firstaid(question)

    print("\nPredicted Category:")
    print(result["category"])

    print("\nConfidence:")
    print(f"{result['confidence']:.2%}")

    print("\nRetrieved Information:")
    print(result.get("context", "None"))

    print("\n" + "=" * 60)
    print("FINAL FIRST-AID ANSWER")
    print("=" * 60)

    print(result["answer"])


if __name__ == "__main__":
    main()
