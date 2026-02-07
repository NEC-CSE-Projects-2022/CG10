# app.py
from flask import Flask, request, render_template, jsonify
from model_loader import load_all_models
from predict import fuse_and_predict
import pandas as pd

app = Flask(__name__)

# Load models once at startup
models, tokenizers, meta_mlp = load_all_models()

# ---------- PAGE ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/images")
def images():
    return render_template("images.html")

@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")

# ---------- BACKEND LOGIC ----------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data and "texts" in data:
        texts = data["texts"]
    else:
        t = request.form.get("text", "")
        texts = [t]

    results = []
    for txt in texts:
        label = fuse_and_predict(txt, models, tokenizers, meta_mlp)
        results.append({"text": txt, "sentiment": label})

    return jsonify({"predictions": results})

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV: {str(e)}"}), 400

    if "text" not in df.columns:
        return jsonify({"error": "CSV must have a 'text' column"}), 400

    results = []
    for txt in df["text"].astype(str).tolist():
        label = fuse_and_predict(txt, models, tokenizers, meta_mlp)
        results.append({"text": txt, "sentiment": label})

    return jsonify({"predictions": results})

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
