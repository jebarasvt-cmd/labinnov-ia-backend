from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import numpy as np
from google import generativeai as genai
import re

app = Flask(__name__)
CORS(app)

# ------------------------
# Config Gemini
# ------------------------
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# ------------------------
# Config stockage
# ------------------------
config_data = {
    "api_key": None,
    "tps_file_path": None,
    "qrs_file_path": None
}

QRS_VECTORS_PATH = "/data/qrs_vectors.json"

# ======================
# UTILITAIRES
# ======================
def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_gemini_embedding(text):
    model = "models/embedding-001"
    result = genai.embed_content(model=model, content=text)
    return result["embedding"]

def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def prepare_qrs_embeddings():
    qrs_data = load_json(config_data["qrs_file_path"])
    if not qrs_data:
        return
    qrs_vectors = []
    for item in qrs_data:
        vec = get_gemini_embedding(item["question"])
        qrs_vectors.append({
            "question": item["question"],
            "réponse": item["réponse"],
            "embedding": vec
        })
    save_json(QRS_VECTORS_PATH, qrs_vectors)

def find_best_qr_semantic(question):
    qrs_vectors = load_json(QRS_VECTORS_PATH)
    if not qrs_vectors:
        return None
    question_vec = get_gemini_embedding(question)
    best_match = None
    best_score = 0
    for item in qrs_vectors:
        score = cosine_similarity(question_vec, item["embedding"])
        if score > best_score:
            best_score = score
            best_match = item
    return best_match if best_score >= 0.80 else None

def reformulate_qr_answer(question, original_answer):
    prompt = f"""
    Tu es LabInnov IA, un assistant éducatif en Sciences de la Vie et de la Terre (SVT).

    On t’a fourni une question d'élève et une réponse brute issue d'une base de données éducative.

    🎯 Ta mission :
    - Reformuler la réponse pour qu'elle soit claire, pédagogique et adaptée à un élève de niveau secondaire.
    - Expliquer le concept avec des phrases simples mais scientifiques.
    - Si nécessaire, ajouter un exemple ou une précision utile à la compréhension.
    - Éviter toute formulation trop technique sans explication.
    - Ne pas ajouter d'informations non vérifiées.

    ⚠️ Format attendu :
    Réponse reformulée directement, sans mentionner que c’est une reformulation.

    Question :
    {question}

    Réponse initiale :
    {original_answer}
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ======================
# INIT — upload fichiers + embeddings QRS
# ======================
@app.route("/init", methods=["POST"])
def init():
    try:
        tps_file = request.files.get("tps_file")
        qrs_file = request.files.get("qrs_file")

        if not tps_file:
            return jsonify({"error": "Fichier TPs requis"}), 400

        os.makedirs("/data", exist_ok=True)

        # --- Sauvegarde TPs ---
        tps_path = os.path.join("/data", "tps.json")
        tps_file.save(tps_path)
        config_data["tps_file_path"] = tps_path
        print(f"✅ Fichier TPs enregistré : {tps_path} ({os.path.getsize(tps_path)/1024:.2f} Ko)")

        # --- Sauvegarde QRs ---
        if qrs_file:
            qrs_path = os.path.join("/data", "qrs.json")
            qrs_file.save(qrs_path)
            config_data["qrs_file_path"] = qrs_path
            print(f"✅ Fichier QRs enregistré : {qrs_path} ({os.path.getsize(qrs_path)/1024:.2f} Ko)")

            # Génération embeddings QRS
            prepare_qrs_embeddings()
            print("✅ Embeddings QRS générés avec succès")

        # Retourne les infos comme /status
        return jsonify({
            "message": "Initialisation réussie",
            "status": "ready",
            "files": status().json
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# ======================
# STATUS — état détaillé
# ======================
@app.route("/status", methods=["GET"])
def status():
    status_info = {}

    def add_file_info(file_key, file_path):
        if file_path and os.path.exists(file_path):
            data = load_json(file_path)
            status_info[file_key] = {
                "present": True,
                "entries": len(data) if isinstance(data, list) else 1,
                "size_kb": round(os.path.getsize(file_path) / 1024, 2)
            }
        else:
            status_info[file_key] = {"present": False}

    # tps.json
    add_file_info("tps.json", config_data.get("tps_file_path"))

    # qrs.json
    add_file_info("qrs.json", config_data.get("qrs_file_path"))

    # qrs_vectors.json
    add_file_info("qrs_vectors.json", QRS_VECTORS_PATH)

    return jsonify(status_info)

# ======================
# 🔍 Debug - Lister fichiers persistants
# ======================
@app.route("/debug-files", methods=["GET"])
def debug_files():
    try:
        files_list = []
        data_path = "/data"
        if os.path.exists(data_path):
            for file_name in os.listdir(data_path):
                file_path = os.path.join(data_path, file_name)
                if os.path.isfile(file_path):
                    size_kb = os.path.getsize(file_path) / 1024
                    files_list.append({
                        "name": file_name,
                        "size_kb": round(size_kb, 2)
                    })
        return jsonify({
            "files": files_list,
            "tps_file_path": config_data.get("tps_file_path"),
            "qrs_file_path": config_data.get("qrs_file_path")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================
# 3️⃣ ASK — Répondre à l’élève
# ======================
@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json.get("question", "").strip()

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"error": "Clé API manquante"}), 500

        genai.configure(api_key=api_key)

        # 1️⃣ Vérifier QRs par recherche sémantique
        match = None
        if config_data["qrs_file_path"] and os.path.exists(config_data["qrs_file_path"]):
            match = find_best_qr_semantic(question)
            if match:
                reformulated = reformulate_qr_answer(question, match["réponse"])
                return jsonify({"source": "qrs", "answer": reformulated})

        # 2️⃣ Sinon → TP
        if config_data["tps_file_path"] and os.path.exists(config_data["tps_file_path"]):
            with open(config_data["tps_file_path"], "r", encoding="utf-8") as f:
                tps_content = f.read()

            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
            Tu es LabInnov IA, un assistant éducatif en Sciences de la Vie et de la Terre (SVT).
            On t’a fourni un extrait de base de données JSON décrivant un TP.

            Ta mission :
            - Transformer ces données brutes en un protocole expérimental clair, pédagogique et détaillé.
            - Reformuler chaque section pour qu’elle soit compréhensible par un élève.
            - Développer les étapes avec des phrases complètes qui expliquent le pourquoi et le comment.
            - Ajouter des consignes de sécurité ou de bonnes pratiques si pertinentes.
            - Utiliser un langage simple mais scientifique.
            - Ne jamais recopier mot pour mot le JSON, mais t'en inspirer.

            ⚠️ Format Markdown obligatoire :
            **Titre :** (reprendre et reformuler le champ "titre")

            **Objectif :** (développer le champ "objectif" en expliquant l’importance de ce TP)

            **Prérequis :**
            - (reformuler chaque prérequis en phrase complète)

            **Matériel :**
            - (lister le matériel, ajouter la fonction ou l’usage de chaque élément entre parenthèses)

            **Procédure expérimentale :**
            1. (reprendre chaque étape et la développer en une ou deux phrases complètes, expliquer le but)
            2. ...
            3. ...

            **Résultats attendus :**
            - (expliquer clairement ce que l’élève doit observer et ce que cela prouve)

            ⚠️ Règles :
            - Minimum 2 phrases complètes par étape.
            - Présentation structurée avec titres, listes à puces et numéros.
            - Ne pas écrire "Protocole expérimental".
            - Respecter la mise en forme Markdown.
            - Toujours commencer par **Titre :**.

            Question de l’élève :
            {question}

            Données disponibles (JSON) :
            {tps_content}
            """
            response = model.generate_content(prompt)
            if not response or not hasattr(response, "text") or not response.text:
                return jsonify({"error": "Réponse vide de Gemini"}), 500

            response_text = re.sub(r"^\**\s*protocole\s+expérimental\s*\**\s*\n*", "", response.text.strip(), flags=re.IGNORECASE)
            final_answer = response_text
            return jsonify({"source": "tps", "answer": final_answer})

        return jsonify({"source": "none", "answer": "Aucune donnée disponible."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================
# 4️⃣ Home test
# ======================
@app.route("/", methods=["GET"])
def home():
    return "✅ LabInnov IA Backend est en ligne sur Render"

# ======================
# Recharge fichiers si déjà présents
# ======================
if os.path.exists("/data/tps.json"):
    config_data["tps_file_path"] = "/data/tps.json"
    print("✅ Fichier TPs rechargé :", config_data["tps_file_path"])

if os.path.exists("/data/qrs.json"):
    config_data["qrs_file_path"] = "/data/qrs.json"
    print("✅ Fichier QRs rechargé :", config_data["qrs_file_path"])
    prepare_qrs_embeddings()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
