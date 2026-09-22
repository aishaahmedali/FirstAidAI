import chromadb

client = chromadb.PersistentClient(
    path="./vector_db"
)

collection = client.get_collection("first_aid")

print("ChromaDB collection:", collection.name)
print("Total stored documents:", collection.count())

# Get stored documents
data = collection.get(
    include=["documents"]
)

documents = data["documents"]

print("\n" + "=" * 70)
print("STORED FIRST-AID DOCUMENTS")
print("=" * 70)

for i, document in enumerate(documents[:20], 1):
    print(f"\nDOCUMENT {i}")
    print("-" * 70)
    print(document[:1000])