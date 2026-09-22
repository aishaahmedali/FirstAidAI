import chromadb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB = BASE_DIR / "vector_db"

client = chromadb.PersistentClient(path=str(VECTOR_DB))

collection = client.get_collection("first_aid")

print("\n" + "=" * 70)
print("CHROMADB FIRST-AID CHUNKS")
print("=" * 70)

print(f"Collection: {collection.name}")
print(f"Total chunks: {collection.count()}")

data = collection.get(
    limit=10,
    include=["documents", "metadatas"]
)

print("\n" + "=" * 70)
print("FIRST 10 STORED CHUNKS")
print("=" * 70)

for i, document in enumerate(data["documents"], 1):
    print(f"\n--- CHUNK {i} ---")
    print(document)

    if data["metadatas"]:
        print(f"Metadata: {data['metadatas'][i-1]}")

print("\n" + "=" * 70)