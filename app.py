from flask import Flask, request, jsonify
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)  # Autorise toutes les origines pour éviter les erreurs CORS

# État du système (simulé pour l'instant)
status_data = {
    "ready": False,
    "last_init_time": None,
    "has_api_key": False
}

@app.route("/")
def home():
    return "LabInnov IA Backend is running on Render!"

@app.route("/init", methods=["POST", "GET"])
def init():
    """
    Simule l'initialisation : stockage de la clé API et chargement des fichiers.
    """
    # Si une clé API est envoyée dans le formulaire
    if request.method == "POST":
        if "api_key" in request.form:
            api_key = request.form["api_key"]
            if api_key.strip():
                status_data["has_api_key"] = True

        # On pourrait aussi traiter ici les fichiers envoyés (TPs, Q&R)
        # tps_file = request.files.get("tps_file")
        # qrs_file = request.files.get("qrs_file")

    status_data["ready"] = True
    status_data["last_init_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    return jsonify({
        "message": "Initialisation terminée",
        "status": "ready",
        "timestamp": status_data["last_init_time"]
    })

@app.route("/status", methods=["GET"])
def status():
    """
    Retourne l'état du système dans le format attendu par le frontend.
    """
    return jsonify({
        "rag_initialized": status_data["ready"],
        "has_cached_api_key": status_data["has_api_key"],
        "has_cached_database": status_data["ready"],
        "fully_configured": status_data["ready"] and status_data["has_api_key"]
    })

@app.route("/ask", methods=["POST"])
def ask():
    """
    Répond à une question envoyée par le frontend.
    Pour l'instant, la réponse est simulée.
    """
    data = request.get_json()
    question = data.get("question", "")

    # Réponse simulée — à remplacer par ton vrai modèle IA
    answer = f"Vous avez demandé : {question} (réponse générée par LabInnov IA)"

    return jsonify({"answer": answer})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
