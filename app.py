from flask import Flask, request, jsonify
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)  # Autoriser toutes les origines (pour ton frontend)

# État simulé de l'application
status_data = {
    "ready": False,
    "last_init_time": None
}

@app.route("/")
def home():
    return "LabInnov IA Backend is running on Render!"

@app.route("/init", methods=["POST", "GET"])
def init():
    """
    Simule une initialisation du modèle / chargement des données
    """
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
    Retourne l'état actuel du backend (prêt ou non)
    """
    return jsonify({
        "ready": status_data["ready"],
        "last_init_time": status_data["last_init_time"]
    })

@app.route("/ask", methods=["POST"])
def ask():
    """
    Traite une question envoyée par le frontend
    """
    data = request.get_json()
    question = data.get("question", "")

    # Réponse simulée pour test
    answer = f"Vous avez demandé : {question} (réponse générée par LabInnov IA)"
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
