from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# ======================
# Configuration initiale
# ======================
config_data = {
    "api_key": os.environ.get("GEMINI_API_KEY"),  # ✅ Chargée depuis Render
    "tps_file_path": None,
    "qrs_file_path": None
}

# ======================
# 1️⃣ INIT — Upload fichiers
# ======================
@app.route("/init", methods=["POST"])
def init():
    try:
        tps_file = request.files.get("tps_file")
        qrs_file = request.files.get("qrs_file")

        print("📩 /init appelé")
        print("➡️ Clé API en mémoire :", bool(config_data["api_key"]))
        print("➡️ Fichier TPs reçu :", tps_file.filename if tps_file else None)

        if not config_data["api_key"]:
            return jsonify({"error": "Clé API absente dans l'environnement"}), 400
        if not tps_file:
            return jsonify({"error": "Fichier TPs requis"}), 400

        # Sauvegarder les fichiers uploadés
        os.makedirs("uploaded_files", exist_ok=True)
        tps_path = os.path.join("uploaded_files", tps_file.filename)
        tps_file.save(tps_path)
        config_data["tps_file_path"] = tps_path

        if qrs_file:
            qrs_path = os.path.join("uploaded_files", qrs_file.filename)
            qrs_file.save(qrs_path)
            config_data["qrs_file_path"] = qrs_path

        print("✅ Configuration enregistrée avec succès")
        return jsonify({"message": "Initialisation réussie", "status": "ready"}), 200

    except Exception as e:
        print("❌ Erreur /init :", str(e))
        return jsonify({"error": str(e)}), 500


# ======================
# 2️⃣ STATUS — État du système
# ======================
@app.route("/status", methods=["GET"])
def status():
    fully_configured = bool(config_data["api_key"] and config_data["tps_file_path"])
    return jsonify({
        "rag_initialized": fully_configured,
        "has_cached_api_key": bool(config_data["api_key"]),
        "has_cached_database": bool(config_data["tps_file_path"]),
        "fully_configured": fully_configured
    })


# ======================
# 3️⃣ ASK — Génération avec Gemini
# ======================
@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json.get("question")

        if not config_data["api_key"]:
            return jsonify({"error": "Clé API manquante"}), 400
        if not config_data["tps_file_path"]:
            return jsonify({"error": "Aucun fichier TPs disponible"}), 400

        # Charger le fichier TPs
        with open(config_data["tps_file_path"], "r", encoding="utf-8") as f:
            tps_content = f.read()

        # Configurer Gemini
        genai.configure(api_key=config_data["api_key"])
        model = genai.GenerativeModel("gemini-1.5-flash")  # ⚡ Plus rapide

        # Prompt
        prompt = f"""
Tu es LabInnov IA, un assistant éducatif en Sciences de la Vie et de la Terre (SVT).
On t’a fourni un extrait de base de données JSON décrivant un TP, avec les champs :
"titre", "objectif", "prérequis", "matériel", "étapes", "résultats_attendus".

Données disponibles :
{tps_content}

Tâche :
- Utilise uniquement les informations fournies pour rédiger un **protocole expérimental complet**.
- Ce protocole doit guider l'élève pas à pas pour réaliser le TP.
- Structure le texte avec les sections suivantes :
  1. **Titre**
  2. **Objectif**
  3. **Prérequis**
  4. **Matériel**
  5. **Procédure expérimentale**
  6. **Résultats attendus**

Question de l’élève :
{question}
"""

        print("📤 Envoi du prompt à Gemini Flash...")
        response = model.generate_content(prompt)
        print("📥 Réponse reçue de Gemini Flash")

        return jsonify({"answer": response.text})

    except Exception as e:
        print("❌ Erreur /ask :", str(e))
        return jsonify({"error": str(e)}), 500


# ======================
# 4️⃣ ROUTE DE TEST
# ======================
@app.route("/", methods=["GET"])
def home():
    return "✅ LabInnov IA Backend est en ligne sur Render (Gemini Flash)"


# ======================
# Lancement local
# ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
