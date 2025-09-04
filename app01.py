from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Autorise les requêtes depuis ton frontend

@app.route("/")
def home():
    return "LabInnov IA Backend is running!"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    # Ici, logiquement, tu appelleras ton IA / RAG
    return jsonify({"answer": f"Vous avez demandé : {question}"})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
