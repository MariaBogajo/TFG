from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
COLLECTION = "rag_demo"

def main():
    client = QdrantClient(url=QDRANT_URL)

    # Borrar la colección si existe
    if client.collection_exists(COLLECTION):
        client.delete_collection(collection_name=COLLECTION)
        print(f"🧹 Colección borrada: {COLLECTION}")
    else:
        print(f"ℹ️ La colección no existe: {COLLECTION}")

if __name__ == "__main__":
    main()