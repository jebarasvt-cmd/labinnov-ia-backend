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
        tps_file = request.files.get("tps_file")
        qrs_file = request.files.get("qrs_file")

        if not tps_file:
            return jsonify({"error": "Fichier TPs requis"}), 400

        # 📌 Sauvegarder dans le disque persistant
        os.makedirs("/data", exist_ok=True)

        # Sauvegarder le fichier TPs
        tps_path = os.path.join("/data", "tps.json")
        tps_file.save(tps_path)
        config_data["tps_file_path"] = tps_path

        # Sauvegarder le fichier QRs si présent
        if qrs_file:
            qrs_path = os.path.join("/data", "qrs.json")
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
    fully_configured = bool(config_data["tps_file_path"])
    return jsonify({
        "rag_initialized": fully_configured,
        "has_cached_api_key": bool(os.environ.get("API_KEY")),
        "has_cached_database": bool(config_data["tps_file_path"]),
        "fully_configured": fully_configured
    })

# ======================
# 🔍 Debug - Lister les fichiers persistants
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
# 2️⃣ STATUS — État du système
# ======================
@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json.get("question")

        if not os.environ.get("API_KEY"):
            return jsonify({"error": "Clé API manquante dans les variables d'environnement"}), 500
        if not config_data.get("tps_file_path"):
            return jsonify({"error": "Aucun fichier TPs disponible"}), 400

        # Charger le fichier TPs
        with open(config_data["tps_file_path"], "r", encoding="utf-8") as f:
            tps_content = f.read()

        # Initialiser Gemini avec la clé depuis l'environnement
        genai.configure(api_key=os.environ.get("API_KEY"))
        model = genai.GenerativeModel("gemini-pro")

        prompt = f"""
Tu es LabInnov IA, un assistant éducatif en Sciences de la Vie et de la Terre (SVT).
On t’a fourni un extrait de base de données JSON décrivant un TP.

⚠️ Format de réponse OBLIGATOIRE (respecter les retours à la ligne et la mise en forme Markdown) :

**Protocole expérimental**

**Titre :** ...
**Objectif :** ...
**Prérequis :** ...
**Matériel :**
- ...
- ...
**Procédure expérimentale :**
1. ...
2. ...
**Résultats attendus :**
- ...
- ...

Question de l’élève :
{question}

Données disponibles :
{tps_content}
"""

        response = model.generate_content(prompt)

        # Forcer le titre en gras au début
        answer = "**Protocole expérimental**\n\n" + response.text.strip()

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500




# ======================
# 4️⃣ ROUTE DE TEST
# ======================
@app.route("/", methods=["GET"])
def home():
    return "✅ LabInnov IA Backend est en ligne sur Render"

# ======================
# 0️⃣ Recharge automatique de la config si fichiers déjà présents
# ======================
if os.path.exists("/data/tps.json"):
    config_data["tps_file_path"] = "/data/tps.json"
    print("✅ Fichier TPs trouvé et rechargé :", config_data["tps_file_path"])

if os.path.exists("/data/qrs.json"):
    config_data["qrs_file_path"] = "/data/qrs.json"
    print("✅ Fichier QRs trouvé et rechargé :", config_data["qrs_file_path"])

# ======================
# Lancement local
# ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
