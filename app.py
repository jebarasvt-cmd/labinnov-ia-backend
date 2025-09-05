from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from google import generativeai as genai

app = Flask(__name__)
CORS(app)

# Stockage minimal pour économiser la mémoire sur Render Free Tier
config_data = {
    "api_key": None,
    "tps_file_path": None,
    "qrs_file_path": None
}

# ======================
# 1️⃣ INIT — Configuration IA + Upload fichiers
# ======================
@app.route("/init", methods=["POST"])
def init():
    try:
        api_key = request.form.get("api_key")
        tps_file = request.files.get("tps_file")
        qrs_file = request.files.get("qrs_file")

        if not api_key or not tps_file:
            return jsonify({"error": "Clé API et fichier TPs requis"}), 400

        # Stocker la clé API
        config_data["api_key"] = api_key

        # Sauvegarder les fichiers uploadés
        os.makedirs("uploaded_files", exist_ok=True)
        tps_path = os.path.join("uploaded_files", tps_file.filename)
        tps_file.save(tps_path)
        config_data["tps_file_path"] = tps_path

        if qrs_file:
            qrs_path = os.path.join("uploaded_files", qrs_file.filename)
            qrs_file.save(qrs_path)
            config_data["qrs_file_path"] = qrs_path

        return jsonify({"message": "Initialisation réussie", "status": "ready"}), 200

    except Exception as e:
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
# 3️⃣ ASK — Génération avec ton prompt fixe
# ======================
@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json.get("question")

        if not config_data["api_key"]:
            return jsonify({"error": "Serveur non configuré"}), 400
        if not config_data["tps_file_path"]:
            return jsonify({"error": "Aucun fichier TPs disponible"}), 400

        # Charger le fichier TPs
        with open(config_data["tps_file_path"], "r", encoding="utf-8") as f:
            tps_content = f.read()

        # Initialiser Gemini uniquement ici
        genai.configure(api_key=config_data["api_key"])
        model = genai.GenerativeModel("gemini-pro")

        # 📌 PROMPT EXACT
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
  5. **Procédure expérimentale** (reprend et développe les étapes en phrases complètes et numérotées)
  6. **Résultats attendus**

Règles :
- Ne pas inventer de contenu absent des données.
- Adapter le vocabulaire au niveau scolaire.
- Écrire de manière claire, concise et motivante.

Question de l’élève :
{question}

Format attendu (texte structuré) :

**Titre :** ...
**Objectif :** ...
**Prérequis :** ...
**Matériel :**
- ...
**Procédure expérimentale :**
1. ...
2. ...
**Résultats attendus :**
...
"""

        # Envoyer le prompt à Gemini
        response = model.generate_content(prompt)

        return jsonify({"answer": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ======================
# 4️⃣ ROUTE DE TEST
# ======================
@app.route("/", methods=["GET"])
def home():
    return "✅ LabInnov IA Backend est en ligne sur Render"


# ======================
# Lancement local
# ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
