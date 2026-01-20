from dotenv import load_dotenv

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

COLLECTION = "rag_demo"
QDRANT_URL = "http://localhost:6333"


def main():
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
    )

    while True:
        q = input("\nPregunta (ENTER para salir): ").strip()
        if not q:
            break

        resp = query_engine.query(q)

        print("\n=== RESPUESTA ===")
        print(resp.response)

        print("\n=== FRAGMENTOS RECUPERADOS (EVIDENCIAS) ===")
        for i, node in enumerate(resp.source_nodes, start=1):
            meta = node.node.metadata
            src = meta.get("file_name", "¿?")
            print(f"\n[{i}] Fuente: {src}")
            text = node.node.get_text()
            print(text[:700] + (" ..." if len(text) > 700 else ""))


if __name__ == "__main__":
    main()