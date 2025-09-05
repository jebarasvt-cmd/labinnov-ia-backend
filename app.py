import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# -------------------- VARIABLES GLOBALES --------------------
documents = []
embeddings = None
status_data = {
    "rag_initialized": False,
    "has_cached_api_key": False,
    "has_cached_database": False,
    "fully_configured": False
}

# -------------------- FONCTIONS RAG --------------------
def create_embeddings(texts):
    """
    Crée les embeddings pour une liste de textes.
    """
    model = "models/text-embedding-004"
    vectors = []
    for text in texts:
        try:
            emb = genai.embed_content(model=model, content=text)
            vectors.append(emb["embedding"])
        except Exception as e:
            print(f"Erreur embedding: {e}")
            vectors.append([0.0] * 768)  # vecteur neutre en cas d'erreur
    return np.array(vectors)


def load_data(tp_file, qr_file=None):
    """
    Charge les données depuis les fichiers JSON TP et Q&R et calcule les embeddings.
    """
    global documents, embeddings

    documents = []

    # Charger TPs
    with open(tp_file, "r", encoding="utf-8") as f:
        tp_data = json.load(f)
        for item in tp_data:
            text = f"TP: {item.get('titre', '')} - {item.get('description', '')} - Questions: {item.get('questions', '')}"
            documents.append(text)

    # Charger Q&R
    if qr_file:
        with open(qr_file, "r", encoding="utf-8") as f:
            qr_data = json.load(f)
            for qa in qr_data:
                text = f"Q: {qa.get('question', '')} - R: {qa.get('reponse', '')}"
                documents.append(text)

    # Créer embeddings
    embeddings = create_embeddings(documents)


def search(query, top_n=3):
    """
    Recherche vectorielle par similarité cosinus.
    """
    query_emb = create_embeddings([query])
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_n]
    return [documents[i] for i in top_indices]


def generate_with_gemini(context, question):
    """
    Génère la réponse finale avec Gemini Pro à partir du contexte et de la question.
    """
    prompt = f"""
Tu es LabInnov IA, un assistant éducatif en SVT.
Voici des documents pertinents extraits de la base :
{context}

Réponds à la question suivante de façon claire, pédagogique et adaptée à un élève :
{question}
"""
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

# -------------------- ROUTES --------------------
@app.route("/status", methods=["GET"])
def status():
    return jsonify(status_data)


@app.route("/init", methods=["POST"])
def init_system():
    try:
        api_key = request.form.get("api_key")
        tp_file = request.files.get("tps_file")
        qr_file = request.files.get("qrs_file")

        if not api_key:
            return jsonify({"error": "Clé API Gemini requise"}), 400

        if not tp_file:
            return jsonify({"error": "Fichier TPs requis"}), 400

        # Sauvegarder fichiers temporairement
        tp_path = os.path.join("/tmp", tp_file.filename)
        tp_file.save(tp_path)

        qr_path = None
        if qr_file:
            qr_path = os.path.join("/tmp", qr_file.filename)
            qr_file.save(qr_path)

        # Configurer Gemini
        genai.configure(api_key=api_key)
        status_data["has_cached_api_key"] = True

        # Charger données et embeddings
        load_data(tp_path, qr_path)
        status_data["has_cached_database"] = True
        status_data["rag_initialized"] = True
        status_data["fully_configured"] = True

        return jsonify({"message": "Initialisation terminée", "status": "ready"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        if not status_data["fully_configured"]:
            return jsonify({"error": "Le système n'est pas encore configuré"}), 400

        question = request.json.get("question")
        if not question:
            return jsonify({"error": "Question requise"}), 400

        # Recherche vectorielle
        docs = search(question)
        context = "\n".join(docs)

        # Génération IA
        answer = generate_with_gemini(context, question)

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- LANCEMENT --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
