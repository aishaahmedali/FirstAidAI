from pathlib import Path
import chromadb

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_FILE = BASE_DIR / "documents" / "first_aid_rag_source.txt"

CHUNK_SIZE = 1000
OVERLAP = 200

with open(SOURCE_FILE, "r", encoding="utf-8") as file:
    text = file.read()

chunks = []
start = 0

while start < len(text):
    end = start + CHUNK_SIZE
    chunk = text[start:end].strip()

    if chunk:
        chunks.append(chunk)

    start += CHUNK_SIZE - OVERLAP

client = chromadb.PersistentClient(path=str(BASE_DIR / "vector_db"))
collection = client.get_collection(name="first_aid")

stored_count = collection.count()

print("\n========== CHUNKING VERIFICATION ==========")
print("Source document:", SOURCE_FILE.name)
print("Source characters:", len(text))
print("Chunk size:", CHUNK_SIZE)
print("Chunk overlap:", OVERLAP)
print("Chunk step:", CHUNK_SIZE - OVERLAP)
print("Independently generated chunks:", len(chunks))
print("Chunks stored in ChromaDB:", stored_count)

print("\n========== COMPARISON ==========")

if len(chunks) == stored_count:
    print("SUCCESS: Generated and stored chunk counts match!")
else:
    print("WARNING: Generated and stored chunk counts do not match!")

print("\n========== CHUNK LENGTH STATISTICS ==========")
lengths = [len(chunk) for chunk in chunks]

print("Shortest chunk:", min(lengths))
print("Longest chunk:", max(lengths))
print("Average chunk length:", round(sum(lengths) / len(lengths), 2))

print("\n========== TEST COMPLETE ==========")
