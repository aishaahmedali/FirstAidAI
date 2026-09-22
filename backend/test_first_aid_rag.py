import chromadb
from pathlib import Path

VECTOR_DB = Path(__file__).resolve().parent.parent / "vector_db"

client = chromadb.PersistentClient(path=str(VECTOR_DB))

collection = client.get_collection("first_aid")

questions = [
    "What are the steps of emergency scene management?",
    "What should I do if someone is choking?",
    "What should I do for severe bleeding?",
    "When should CPR be given?",
    "What should I do for a burn?"
]

print("=" * 70)
print("FIRST-AID RAG DEMONSTRATION")
print("=" * 70)

print(f"Collection: first_aid")
print(f"Total stored chunks: {collection.count()}")

for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)
    print("=" * 70)

    results = collection.query(
        query_texts=[question],
        n_results=2
    )

    documents = results["documents"][0]

    for i, document in enumerate(documents, 1):

        print(f"\n--- RETRIEVED FIRST-AID CHUNK {i} ---")
        print(document[:1500])

print("\n" + "=" * 70)
print("END OF RAG DEMONSTRATION")
print("=" * 70)
