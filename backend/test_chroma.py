import chromadb

client = chromadb.PersistentClient(path="vector_db")
collection = client.get_collection(name="first_aid")

print("\n========== CHROMADB VERIFICATION ==========")
print("Collection name:", collection.name)
print("Total stored chunks:", collection.count())

data = collection.get(limit=3, include=["documents"])

for i, document in enumerate(data["documents"], 1):
    print(f"\nCHUNK {i}")
    print("Characters:", len(document))
    print("Content:", document[:500])

print("\n========== VERIFICATION COMPLETE ==========")
