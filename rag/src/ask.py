from dotenv import load_dotenv
from importlib.metadata import version as pkg_version
from datetime import datetime, timezone
from pathlib import Path
import json
import argparse
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

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
SCHEMA_VERSION = "v1"


# =========================
#   Salida estructurada
# =========================
class RagAnswer(BaseModel):
    summary: str = Field(..., description="Resumen breve del caso y la orientación general.")
    red_flags: List[str] = Field(default_factory=list, description="Lista de señales de alarma relevantes.")
    priority: Literal["low", "medium", "high"] = Field(..., description="Nivel de prioridad/urgencia.")
    recommendation: str = Field(..., description="Recomendación práctica (sin diagnóstico ni tratamiento).")
    justification: str = Field(..., description="Justificación basada EXCLUSIVAMENTE en evidencias recuperadas.")
    sources: List[str] = Field(default_factory=list, description="Fuentes usadas (paths o ids).")


def build_module_filters(module_value: str) -> MetadataFilters:
    return MetadataFilters(filters=[MetadataFilter(key="module", value=module_value)])


def ensure_log_dir():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def extract_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def safe_parse_answer(raw: str) -> RagAnswer:
    raw_json = extract_json(raw)
    data = json.loads(raw_json)

    # Normalización robusta de priority (por si el modelo responde en español)
    if isinstance(data, dict) and "priority" in data and isinstance(data["priority"], str):
        p = data["priority"].strip().lower()
        mapping = {
            "low": "low", "bajo": "low", "baja": "low",
            "medium": "medium", "medio": "medium", "media": "medium",
            "high": "high", "alto": "high", "alta": "high",
        }
        if p in mapping:
            data["priority"] = mapping[p]

    return RagAnswer.model_validate(data)


def format_evidence(nodes) -> tuple[str, list[str]]:
    lines = []
    sources = []

    for i, n in enumerate(nodes, start=1):
        meta = n.node.metadata or {}
        src = meta.get("source_path") or meta.get("file_name") or "unknown_source"
        text = n.node.get_text() or ""
        lines.append(f"[{i}] SOURCE={src}\n{text}\n")
        sources.append(src)

    seen = set()
    sources_unique = [s for s in sources if not (s in seen or seen.add(s))]
    return "\n".join(lines).strip(), sources_unique


def build_json_prompt(question: str, evidence_text: str, module: str) -> str:
    return f"""
Eres un asistente clínico SIMULADO para un TFG. No haces diagnóstico ni tratamiento.
Tu tarea es generar una orientación basada EXCLUSIVAMENTE en la evidencia proporcionada.

MÓDULO ACTIVO: {module}

EVIDENCIA (fragmentos recuperados):
{evidence_text}

PREGUNTA / CASO:
{question}

Devuelve SOLO un JSON válido (sin markdown ni texto adicional) con esta estructura EXACTA:
{{
  "summary": "string",
  "red_flags": ["string", "..."],
  "priority": "low|medium|high",
  "recommendation": "string",
  "justification": "string",
  "sources": ["string", "..."]
}}

Reglas:
- No inventes información.
- Si algo no consta en la evidencia, indícalo en "justification".
- En "sources" devuelve SOLO los valores que aparecen como SOURCE=... en la evidencia (rutas), no títulos.
- Responde en español.
- Devuelve "red_flags" en español.
- MUY IMPORTANTE: el campo "priority" debe ser EXACTAMENTE uno de estos tres valores en inglés: "low", "medium" o "high".
""".strip()


def _unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def log_interaction(
    module: str,
    question: str,
    answer_text: str,
    source_nodes,
    top_k: int,
    retrieval_count: int,
    retrieved_sources: List[str],
    scores: List[Optional[float]],
    answer_json=None,
    answer_raw=None,
):
    ensure_log_dir()
    record = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "collection": COLLECTION,
        "module": module,
        "question": question,
        "top_k": top_k,
        "retrieval_count": retrieval_count,
        "retrieved_sources": retrieved_sources,
        "scores": scores,
        "answer_text": answer_text,
        "answer_json": answer_json,
        "answer_raw": answer_raw,
        "schema_version": SCHEMA_VERSION,
        "chunks": [],
    }

    for n in source_nodes:
        meta = n.node.metadata or {}
        record["chunks"].append(
            {
                "module": meta.get("module"),
                "source_path": meta.get("source_path", meta.get("file_name")),
                "score": getattr(n, "score", None),
                "text_preview": (n.node.get_text() or "")[:700],
            }
        )

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG demo con Qdrant + LlamaIndex + salida estructurada (1 llamada al LLM)"
    )
    parser.add_argument(
        "--module",
        default="triage",
        choices=["triage", "ap", "derivation"],
        help="Módulo de la base de conocimiento.",
    )
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def suggest_module_warning(question: str, module_active: str) -> str | None:
    q = (question or "").lower()

    ap_keywords = [
        "primary care", "general practice", "family doctor", "gp",
        "atención primaria", "medicina de familia", "centro de salud",
        "follow-up", "follow up", "followup", "safety-net", "safety net",
        "review in", "routine follow-up", "appointment", "consulta", "cita",
        "warning signs", "signos de alarma", "derivación", "derivar", "referral"
    ]

    # OJO: quitamos "urgent/urgency" porque es ambiguo (AP también habla de urgente)
    triage_keywords = [
        "triage", "triaje", "ed", "er",
        "emergency", "emergencia", "emergency department",
        "resuscitation", "reanimación",
        "life-threatening", "life threatening", "amenaza vital",
        "immediate assessment", "evaluación inmediata",
        "stroke", "ictus", "fast"
    ]

    ap_score = sum(1 for k in ap_keywords if k in q)
    triage_score = sum(1 for k in triage_keywords if k in q)

    # Aviso solo si hay una diferencia clara
    if module_active == "triage" and ap_score >= triage_score + 1:
        return "⚠️ Tu pregunta suena a Atención Primaria. Si quieres, prueba: python src/ask.py --module ap"

    if module_active == "ap" and triage_score >= ap_score + 1:
        return "⚠️ Tu pregunta suena a triaje/urgencias. Si quieres, prueba: python src/ask.py --module triage"

    return None


def main():
    args = parse_args()
    module_active = args.module
    top_k = args.top_k

    print("ℹ️ llama_index version:", pkg_version("llama-index"))
    print(f"ℹ️ Módulo activo: {module_active}")
    print(f"ℹ️ top_k: {top_k}")

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
    filters = build_module_filters(module_active)

    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=filters,
    )

    while True:
        q = input("\nPregunta (ENTER para salir): ").strip()
        if not q:
            break

        warning = suggest_module_warning(q, module_active)
        if warning:
            print(f"\n{warning}")

        nodes = retriever.retrieve(q)

        retrieval_count = len(nodes)
        sources_raw = []
        scores = []
        for n in nodes:
            meta = n.node.metadata or {}
            sources_raw.append(meta.get("source_path") or meta.get("file_name") or "unknown_source")
            scores.append(getattr(n, "score", None))
        retrieved_sources = _unique_keep_order(sources_raw)

        print("\n=== INFO ===")
        print(f"🔎 Módulo usado: {module_active}")
        print(f"🔎 Chunks recuperados: {retrieval_count}")

        if not nodes:
            msg = (
                f"No hay evidencias en el módulo '{module_active}'.\n"
                f"Añade documentos a data/{module_active}/ y reindexa."
            )
            print(f"\n⚠️ {msg}")
            log_interaction(
                module_active,
                q,
                msg,
                [],
                top_k=top_k,
                retrieval_count=retrieval_count,
                retrieved_sources=retrieved_sources,
                scores=scores,
                answer_json=None,
                answer_raw=None,
            )
            print(f"\n📝 Guardado log en: {LOG_PATH.as_posix()}")
            continue

        evidence_text, sources = format_evidence(nodes)
        prompt = build_json_prompt(q, evidence_text, module_active)

        raw = llm.complete(prompt).text

        try:
            parsed = safe_parse_answer(raw)
        except (json.JSONDecodeError, ValidationError):
            repair = f"""
Devuelve SOLO un JSON válido (sin texto extra) que cumpla exactamente el schema, a partir de este contenido:
{raw}
""".strip()
            raw = llm.complete(repair).text
            parsed = safe_parse_answer(raw)

        # ✅ SIEMPRE forzamos las fuentes reales (no las del modelo)
        parsed.sources = sources

        answer_json = parsed.model_dump()
        answer_json["module"] = module_active
        answer_json["top_k"] = top_k
        answer_json["retrieval_count"] = retrieval_count
        answer_json["retrieved_sources"] = retrieved_sources

        answer_text = (
            f"{parsed.summary}\n\n"
            f"priority: {parsed.priority}\n"
            f"red_flags: {', '.join(parsed.red_flags) or 'N/A'}\n\n"
            f"recommendation: {parsed.recommendation}\n\n"
            f"justification: {parsed.justification}\n"
            f"sources: {', '.join(parsed.sources)}"
        )

        print("\n=== RESPUESTA (JSON) ===")
        print(json.dumps(answer_json, indent=2, ensure_ascii=False))

        print("\n=== FRAGMENTOS RECUPERADOS (EVIDENCIAS) ===")
        for i, n in enumerate(nodes, start=1):
            meta = n.node.metadata or {}
            src = meta.get("source_path", meta.get("file_name", "¿?"))
            score = getattr(n, "score", None)

            print(f"\n[{i}] Fuente: {src}")
            if score is not None:
                print(f"    Score: {score}")
            print((n.node.get_text() or "")[:700])

        log_interaction(
            module_active,
            q,
            answer_text,
            nodes,
            top_k=top_k,
            retrieval_count=retrieval_count,
            retrieved_sources=retrieved_sources,
            scores=scores,
            answer_json=answer_json,
            answer_raw=raw,
        )

        print(f"\n📝 Guardado log en: {LOG_PATH.as_posix()}")


if __name__ == "__main__":
    main()