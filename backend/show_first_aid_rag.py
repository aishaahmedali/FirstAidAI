import chromadb

client = chromadb.PersistentClient(path="./vector_db")

collection = client.get_collection("first_aid")

print("=" * 80)
print("FIRST AID RAG KNOWLEDGE BASE")
print("Source: St. John Ambulance Canada First Aid Reference Guide")
print("=" * 80)

data = collection.get(include=["documents", "metadatas"])

documents = data["documents"]

topics = [
    "Burns",
    "Bleeding",
    "Choking",
    "CPR",
    "Bites and stings",
    "Fractures",
    "Shock",
    "Poisoning",
    "Temperature emergencies",
    "Neurological emergencies"
]

found = 0

for topic in topics:
    print("\n" + "=" * 80)
    print(f"TOPIC: {topic.upper()}")
    print("=" * 80)

    topic_found = 0

    for i, document in enumerate(documents):

        if topic.lower() in document.lower():

            print(f"\n--- CHUNK {i + 1} ---")
            print(document[:1500])
            print()

            topic_found += 1
            found += 1

            if topic_found >= 2:
                break

    if topic_found == 0:
        print("No matching chunk found.")

print("\n" + "=" * 80)
print(f"TOTAL DISPLAYED CHUNKS: {found}")
print("=" * 80)