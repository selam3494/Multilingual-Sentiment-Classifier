# sentiment/model_io.py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .constants import LABELS, MAX_LEN

def load_model(path="model_en_light_best"):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tok, model, device

def predict_one(text, tok, model, device, thresh=0.6):
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    probs_pct = (probs * 100).round(2)
    return {
        "text": text,
        "prediction": LABELS[pred_idx],
        "confidence_pct": float(probs_pct[pred_idx]),
        "probs_pct": {lab: float(p) for lab, p in zip(LABELS, probs_pct)},
        "confident": float(probs_pct[pred_idx]) >= thresh * 100,
    }

def predict_batch(texts, tok, model, device, thresh=0.6):
    return [predict_one(t, tok, model, device, thresh) for t in texts]
