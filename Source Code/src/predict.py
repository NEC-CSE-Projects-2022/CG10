# predict.py
import torch
import numpy as np
from model_loader import DEVICE
import math

# Helper: apply softmax safely and return numpy array
def softmax_np(logits):
    a = np.array(logits, dtype=float)
    a_max = a.max()
    e = np.exp(a - a_max)
    return (e / e.sum()).astype(float)

def get_model_probs(model, tokenizer, text):
    """
    Returns:
      probs (np.array): shape (num_classes,) being probability per class as model outputs
      raw_logits (np.array)
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.detach().cpu().numpy().flatten()
    probs = softmax_np(logits)
    return probs, logits

def majority_vote_label(per_model_preds):
    # per_model_preds: list of int labels (0..C-1)
    from collections import Counter
    cnt = Counter(per_model_preds)
    label, count = cnt.most_common(1)[0]
    return label

def fuse_and_predict(text, models, tokenizers, meta_mlp, debug=False):
    """
    Safer fusion:
      - Compute each model's class probabilities (shape C)
      - Align number of classes: if a model outputs 2 classes, convert to 3-class vector [neg, neu=0, pos]
      - Average probabilities across models -> avg_probs (3,)
      - Use avg_probs directly as input to meta_mlp (no scaling). If meta_mlp expects other format, try fallback.
      - Provide fallback majority-vote and avg-prob decision.
    """
    per_model_probs = []
    per_model_logits = []
    per_model_preds = []
    model_names = list(models.keys())

    for key in model_names:
        model = models[key]
        tok = tokenizers[key]
        probs, logits = get_model_probs(model, tok, text)
        per_model_logits.append({ "model": key, "logits": logits.tolist() })
        per_model_probs.append({ "model": key, "probs": probs.tolist() })
        per_model_preds.append(int(np.argmax(probs)))

    # Standardize to 3 classes (neg, neutral, pos)
    # If model is binary (2 classes), we assume ordering [neg, pos] -> expand to [neg, 0.0, pos]
    standardized_probs = []
    for entry in per_model_probs:
        p = np.array(entry["probs"], dtype=float)
        if p.size == 3:
            standardized_probs.append(p)
        elif p.size == 2:
            # assume [neg, pos] ordering -> create [neg, 0.0, pos]
            standardized_probs.append(np.array([p[0], 0.0, p[1]], dtype=float))
        else:
            # unexpected class size: try to map last to positive, first to negative and middle to neutral if >3
            if p.size > 3:
                # reduce by summing tails into last class (poor man's reduce)
                neg = p[0]
                pos = p[-1]
                mid = p[1:-1].sum()
                standardized_probs.append(np.array([neg, mid, pos], dtype=float))
            else:
                # fallback: pad with zeros
                padded = np.zeros(3, dtype=float)
                padded[:p.size] = p
                standardized_probs.append(padded)

    # Normalize each standardized probs just in case
    standardized_probs = np.array([p / (p.sum() + 1e-12) for p in standardized_probs])

    # Average across models
    avg_probs = np.mean(standardized_probs, axis=0)

    # Prepare features for meta_mlp: many training setups use avg_probs as 3 features
    X_meta = avg_probs.reshape(1, -1)  # shape (1,3)

    # Try meta_mlp prediction (safe try/except)
    meta_pred = None
    meta_conf = None
    try:
        pred = meta_mlp.predict(X_meta)[0]
        meta_pred = int(pred)
        # If meta_mlp has predict_proba:
        if hasattr(meta_mlp, "predict_proba"):
            meta_conf = float(np.max(meta_mlp.predict_proba(X_meta)))
    except Exception as e:
        # meta classifier failed (likely expects different features) -> leave as None
        meta_pred = None

    # Fallbacks
    # 1) Average-prob decision: argmax(avg_probs)
    avg_decision = int(np.argmax(avg_probs))
    # 2) Majority vote from per-model predictions (after mapping binary preds to 3-class if needed)
    mapped_preds = []
    for p_arr in per_model_probs:
        p = np.array(p_arr["probs"], dtype=float)
        if p.size == 3:
            mapped_preds.append(int(np.argmax(p)))
        elif p.size == 2:
            # assume [neg,pos]
            lp = int(np.argmax(p))
            if lp == 0:
                mapped_preds.append(0)
            else:
                mapped_preds.append(2)
        else:
            mapped_preds.append(int(np.argmax(p) if p.size>0 else 1))

    majority = majority_vote_label(mapped_preds)

    # Map numeric labels to strings (NOTE: depends on your original mapping — adjust if needed)
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    # Decide final label priority:
    # If meta_pred exists -> use it
    # else use avg_decision
    final_label_index = None
    source = ""
    if meta_pred is not None:
        final_label_index = meta_pred
        source = "meta_mlp"
    else:
        final_label_index = avg_decision
        source = "avg_probs"

    # Compose debug info
    debug_info = {
        "text": text,
        "per_model_logits": per_model_logits,
        "per_model_probs": per_model_probs,
        "standardized_probs": [p.tolist() for p in standardized_probs],
        "avg_probs": avg_probs.tolist(),
        "avg_decision": int(avg_decision),
        "majority_vote": int(majority),
        "meta_pred": (int(meta_pred) if meta_pred is not None else None),
        "meta_confidence": meta_conf,
        "final_used_source": source,
        "final_label_index": int(final_label_index),
        "final_label": label_map.get(int(final_label_index), "Unknown")
    }

    # Return either debug or just final label
    if debug:
        return debug_info
    else:
        return debug_info["final_label"]
