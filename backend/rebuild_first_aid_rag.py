import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB = BASE_DIR / "vector_db"
SOURCE_FILE = BASE_DIR / "documents" / "first_aid_rag_source.txt"

COLLECTION_NAME = "first_aid"

print("=" * 70)
print("REBUILDING FIRST-AID CHROMADB")
print("=" * 70)

with open(SOURCE_FILE, "r", encoding="utf-8") as file:
    text = file.read()

print(f"\nSource file: {SOURCE_FILE}")
print(f"Source characters: {len(text):,}")

chunk_size = 1000
overlap = 200

chunks = []

start = 0

while start < len(text):
    end = start + chunk_size
    chunk = text[start:end].strip()

    if chunk:
        chunks.append(chunk)

    start += chunk_size - overlap

print(f"Created chunks: {len(chunks)}")

print("\nLoading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=str(VECTOR_DB))

try:
    client.delete_collection(COLLECTION_NAME)
    print("Old 'first_aid' collection deleted.")
except Exception:
    print("No old 'first_aid' collection found.")

collection = client.create_collection(
    name=COLLECTION_NAME
)

print("New 'first_aid' collection created.")

print("Generating embeddings...")

embeddings = embedding_model.encode(
    chunks,
    show_progress_bar=True
).tolist()

ids = [f"firstaid_chunk_{i}" for i in range(len(chunks))]

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)

print("\n" + "=" * 70)
print("RAG REBUILD COMPLETE")
print("=" * 70)

print(f"Collection: {COLLECTION_NAME}")
print(f"Total chunks: {collection.count()}")

print("\nFIRST 3 STORED CHUNKS")
print("=" * 70)

data = collection.get(
    limit=3,
    include=["documents"]
)

for i, document in enumerate(data["documents"]):
    print(f"\n--- CHUNK {i + 1} ---")
    print(document[:1000])

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)