from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

DATA_DIR = "data"
COLLECTION = "rag_demo"
QDRANT_URL = "http://localhost:6333"


def main():
    # 1) Cargar PDFs
    docs = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()
    if not docs:
        raise RuntimeError(f"No se encontraron documentos en {DATA_DIR}")

    # 2) Chunking
    splitter = SentenceSplitter(chunk_size=800, chunk_overlap=120)

    # 3) Vector DB (Qdrant)
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4) Embeddings
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # 5) Crear índice y persistir en Qdrant
    _ = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        transformations=[splitter],
        embed_model=embed_model,
        show_progress=True,
    )

    print("✅ Indexación completada.")
    print(f"   - Documentos: {len(docs)}")
    print(f"   - Colección Qdrant: {COLLECTION}")
    print(f"   - Qdrant: {QDRANT_URL}")


if __name__ == "__main__":
    main()