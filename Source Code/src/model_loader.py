# model_loader.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path):
    """Load a HuggingFace fine-tuned model and its tokenizer"""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def load_all_models(base_path="models"):
    """Load all base transformer models and the meta-fusion MLP"""
    print("🚀 Loading Transformer models and meta-fusion classifier...")

    models = {}
    tokenizers = {}

    model_dirs = {
        "sentiment140": "Sentiment140_BERT_Full_2epoch",
        "imdb": "IMDB_BERT_2epoch",
        "absa": "ABSA_BERT_3class",
        "xlmr": "xlmr_multilingual_model"
    }

    for key, folder in model_dirs.items():
        path = os.path.join(base_path, folder)
        print(f"→ Loading {key} model from {path}")
        models[key], tokenizers[key] = load_model(path)

    # Load Meta Fusion MLP (sklearn joblib)
    meta_path = os.path.join(base_path, "meta_mlp.pkl")
    meta_mlp = joblib.load(meta_path)
    print("✅ Loaded meta-fusion MLP model.")

    return models, tokenizers, meta_mlp
