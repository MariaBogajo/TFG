from dotenv import load_dotenv
from importlib.metadata import version as pkg_version
from datetime import datetime, timezone
from pathlib import Path
import json
import argparse

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter

load_dotenv()

COLLECTION = "rag_demo"
QDRANT_URL = "http://localhost:6333"
LOG_PATH = Path("logs/queries.jsonl")


def build_module_filters(module_value: str) -> MetadataFilters:
    return MetadataFilters(filters=[MetadataFilter(key="module", value=module_value)])


def ensure_log_dir():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_interaction(module: str, question: str, answer_text: str, source_nodes):
    ensure_log_dir()
    record = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "collection": COLLECTION,
        "module": module,
        "question": question,
        "answer": answer_text,
        "chunks": [],
    }

    for node in source_nodes:
        meta = node.node.metadata or {}
        record["chunks"].append(
            {
                "module": meta.get("module"),
                "source_path": meta.get("source_path", meta.get("file_name")),
                "text_preview": node.node.get_text()[:700],
            }
        )

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="RAG demo (Qdrant + LlamaIndex) con filtrado por módulo.")
    parser.add_argument(
        "--module",
        default="triage",
        choices=["triage", "ap", "derivation"],
        help="Módulo de la base de conocimiento a usar para el retrieval.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Número de chunks a recuperar (similarity_top_k).")
    return parser.parse_args()


def main():
    args = parse_args()
    module_active = args.module
    top_k = args.top_k

    print("ℹ️ llama_index version:", pkg_version("llama-index"))
    print(f"ℹ️ Módulo activo: {module_active}")
    print(f"ℹ️ top_k: {top_k}")

    # Qdrant
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Index desde vector store
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    module_filters = build_module_filters(module_active)

    # Query engine (lo usaremos solo si hay evidencias)
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=top_k,
        filters=module_filters,
    )

    # Retriever (para contar evidencias antes de llamar al LLM)
    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=module_filters,
    )

    while True:
        q = input("\nPregunta (ENTER para salir): ").strip()
        if not q:
            break

        # 1) Recuperación previa
        nodes = retriever.retrieve(q)

        print("\n=== INFO ===")
        print(f"🔎 Módulo usado: {module_active}")
        print(f"🔎 Chunks recuperados: {len(nodes)}")

        # 2) Si no hay evidencias: no generar respuesta “vacía”
        if len(nodes) == 0:
            msg = (
                f"No hay evidencias en el módulo '{module_active}' para responder a la consulta.\n"
                f"Añade documentos a data/{module_active}/ y reindexa."
            )
            print(f"\n⚠️ {msg}")

            log_interaction(module_active, q, msg, [])
            print(f"\n📝 Guardado log en: {LOG_PATH.as_posix()}")
            continue

        # 3) Si hay evidencias, ahora sí generamos respuesta con grounding
        resp = query_engine.query(q)

        print("\n=== RESPUESTA ===")
        print(resp.response)

        print("\n=== FRAGMENTOS RECUPERADOS (EVIDENCIAS) ===")
        for i, node in enumerate(resp.source_nodes, start=1):
            meta = node.node.metadata or {}
            module = meta.get("module", "¿?")
            src = meta.get("source_path", meta.get("file_name", "¿?"))
            print(f"\n[{i}] Módulo: {module}")
            print(f"    Fuente: {src}")

            text = node.node.get_text()
            print(text[:700] + (" ..." if len(text) > 700 else ""))

        log_interaction(module_active, q, resp.response, resp.source_nodes)
        print(f"\n📝 Guardado log en: {LOG_PATH.as_posix()}")


if __name__ == "__main__":
    main()