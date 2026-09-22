import chromadb
from pathlib import Path

VECTOR_DB = Path(__file__).resolve().parent.parent / "vector_db"

client = chromadb.PersistentClient(path=str(VECTOR_DB))
collection = client.get_collection("first_aid")

topics = [
    "burns",
    "choking",
    "bleeding",
    "CPR",
    "fractures",
    "bites and stings"
]

print("=" * 70)
print("FIRST AID RAG — TOPIC RETRIEVAL DEMONSTRATION")
print("=" * 70)
print(f"Collection: {collection.name}")
print(f"Total chunks: {collection.count()}")

for topic in topics:

    print("\n" + "=" * 70)
    print(f"QUERY: {topic}")
    print("=" * 70)

    results = collection.query(
        query_texts=[topic],
        n_results=1
    )

    document = results["documents"][0][0]

    print("\n--- RETRIEVED FIRST-AID CONTENT ---")
    print(document[:1500])

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)