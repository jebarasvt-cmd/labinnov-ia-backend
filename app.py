# app.py
import os, json, traceback
from typing import List, Dict, Any, Tuple
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
from google import generativeai as genai
from google.api_core.exceptions import ResourceExhausted, InvalidArgument, NotFound

# === Config
DATA_DIR = "data"
TPS_JSON_PATH = os.path.join(DATA_DIR, "tps.json")
QRS_JSON_PATH = os.path.join(DATA_DIR, "qrs.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "tps_faiss.index")
CHUNKS_META_PATH = os.path.join(DATA_DIR, "tps_chunks.json")

ST_MODEL_NAME = os.environ.get("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL = os.environ.get("GEN_MODEL", "gemini-1.5-flash")
TOP_K = int(os.environ.get("LABINNOV_TOP_K", 5))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", 6000))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 120))
API_ENV_VARS = ["GOOGLE_API_KEY", "GEMINI_API_KEY", "API_KEY"]

app = Flask(__name__)
CORS(app)

st_model: SentenceTransformer | None = None
faiss_index: faiss.Index | None = None
chunks_meta: List[Dict[str, Any]] = []

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def get_api_key() -> str:
    for name in API_ENV_VARS:
        val = os.environ.get(name)
        if val:
            print(f"🔑 Clé API détectée ({name}): {val[:6]}********")
            return val
    print("⚠️ Aucune clé API trouvée.")
    return ""

def safe_genai_configure():
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("Clé API manquante.")
    genai.configure(api_key=api_key)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def tp_to_text(tp: Dict[str, Any]) -> str:
    parts = []
    for key in ["titre", "title", "objectif", "objectif_pedagogique", "description", "resume", "materiel", "etapes", "resultats", "résultats"]:
        val = tp.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(f"{key}: {val.strip()}")
        elif isinstance(val, list):
            joined = " | ".join(str(x) for x in val)
            if joined.strip():
                parts.append(f"{key}: {joined}")
    if not parts:
        parts.append(json.dumps(tp, ensure_ascii=False))
    return "\n".join(parts)

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def load_st_model():
    global st_model
    if st_model is None:
        print(f"⬇️ Chargement Sentence-Transformers: {ST_MODEL_NAME}")
        st_model = SentenceTransformer(ST_MODEL_NAME)
    return st_model

def build_faiss_index_from_tps() -> Tuple[int, int]:
    ensure_data_dir()
    if not os.path.exists(TPS_JSON_PATH):
        raise FileNotFoundError("tps.json manquant.")
    tps = load_json(TPS_JSON_PATH)
    if not isinstance(tps, list):
        raise ValueError("tps.json doit être une liste.")

    all_chunks: List[Dict[str, Any]] = []
    for tp_id, tp in enumerate(tps):
        title = tp.get("titre") or tp.get("title") or f"TP #{tp_id}"
        full_text = tp_to_text(tp)
        for text in chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP):
            all_chunks.append({"tp_id": tp_id, "title": title, "text": text})

    model = load_st_model()
    texts = [c["text"] for c in all_chunks]
    emb = model.encode(texts, batch_size=64, convert_to_numpy=True)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index = faiss.IndexIDMap2(index)
    ids = np.arange(len(emb)).astype(np.int64)
    index.add_with_ids(emb, ids)

    faiss.write_index(index, FAISS_INDEX_PATH)
    for i, c in enumerate(all_chunks):
        c["id"] = int(i)
    save_json(CHUNKS_META_PATH, all_chunks)

    global faiss_index, chunks_meta
    faiss_index = index
    chunks_meta = all_chunks
    return (len(tps), len(all_chunks))

def load_index_into_memory() -> bool:
    global faiss_index, chunks_meta
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_META_PATH):
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            chunks_meta = load_json(CHUNKS_META_PATH)
            return True
        except Exception as e:
            print("⚠️ Erreur chargement index :", e)
    return False

def search_similar_chunks(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    if faiss_index is None or not chunks_meta:
        raise RuntimeError("Index FAISS non chargé.")
    model = load_st_model()
    q = model.encode([query], convert_to_numpy=True)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    D, I = faiss_index.search(q.astype(np.float32), top_k)
    hits = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        meta = chunks_meta[int(idx)]
        hits.append({
            "id": int(idx),
            "score": float(score),
            "title": meta.get("title", ""),
            "text": meta.get("text", ""),
            "tp_id": meta.get("tp_id", -1)
        })
    return hits

def build_context_block(hits: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    blocks, used = [], 0
    for h in hits:
        block = f"### {h['title']}\n{h['text']}\n"
        if used + len(block) <= max_chars:
            blocks.append(block)
            used += len(block)
        else:
            break
    return "\n".join(blocks)

def file_info(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return {"present": False, "entries": 0, "size_kb": 0.0}
        size_kb = round(os.path.getsize(path) / 1024, 2)
        entries = 0
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    entries = len(data)
                elif isinstance(data, dict):
                    entries = len(data.keys())
        return {"present": True, "entries": entries, "size_kb": size_kb}
    except Exception as e:
        return {"present": False, "error": str(e)}


@app.route("/status", methods=["GET"])
def status():
    try:
        tps_info = file_info(TPS_JSON_PATH)
        qrs_info = file_info(QRS_JSON_PATH)
        qrs_vec_info = file_info(os.path.join(DATA_DIR, "qrs_vectors.json"))

        fully_configured = tps_info.get("present", False)

        return jsonify({
            "system_ready": True,          # 👈 toujours présent pour le front
            "fully_configured": fully_configured,  # 👈 état interne
            "tps.json": tps_info,
            "qrs.json": qrs_info,
            "qrs_vectors.json": qrs_vec_info
        })
    except Exception as e:
        return jsonify({"system_ready": False, "error": str(e)}), 500



@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Question manquante"}), 400

        try:
            safe_genai_configure()
        except Exception as e:
            return jsonify({"error": f"Erreur configuration Gemini : {e}"}), 500

        global faiss_index, chunks_meta
        if faiss_index is None or not chunks_meta:
            if not load_index_into_memory():
                build_faiss_index_from_tps()

        hits = search_similar_chunks(question, TOP_K)
        context = build_context_block(hits, MAX_CONTEXT_CHARS)

        model = genai.GenerativeModel(GEN_MODEL)
        prompt = f"""
Tu es LabInnov IA, un assistant éducatif en SVT. 
Réponds en français de façon claire et structurée.

CONNAISSANCES (extraits de TPs pertinents) :
{context}

CONSIGNE :
- Transforme/ajuste ce contexte en un protocole expérimental bien présenté (Markdown) sans copier mot à mot.
- Structure attendue :
  **Titre :**
  **Objectif :**
  **Prérequis :**
  **Matériel :**
  **Procédure expérimentale :** (au moins 2 phrases par étape)
  **Résultats attendus :**
- Langage simple mais scientifique, ajoute les consignes de sécurité si pertinent.

Question de l'élève :
{question}
"""
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        if not text:
            return jsonify({"error": "Réponse vide du modèle."}), 502
        return jsonify({"answer": text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    ensure_data_dir()
    load_st_model()  # Préchargement
    load_index_into_memory()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
