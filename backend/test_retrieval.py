import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="vector_db")
collection = client.get_collection(name="first_aid")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("\n========== CHROMADB RETRIEVAL TEST ==========")

while True:
    question = input("\nEnter a first-aid question (or exit): ")

    if question.lower() == "exit":
        break

    embedding = model.encode(question)

    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=3,
        include=["documents", "distances"]
    )

    for i, document in enumerate(results["documents"][0]):
        print(f"\nCHUNK {i + 1}")
        print("Distance:", results["distances"][0][i])
        print("Content:\n", document)

print("\nRETRIEVAL TEST FINISHED")
