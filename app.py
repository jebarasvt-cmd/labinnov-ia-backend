from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"system_ready": True, "message": "✅ Backend en ligne et prêt"})

@app.route("/", methods=["GET"])
def home():
    return "✅ Backend minimal fonctionne !"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
