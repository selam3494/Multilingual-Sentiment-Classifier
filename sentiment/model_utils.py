# sentiment/model_utils.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification
from .constants import NUM_LABELS, MAX_LEN

# -------- training side helpers --------
def build_model(model_name: str, num_labels: int = NUM_LABELS):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def freeze_last_n_layers(model, base_attr="distilbert.transformer.layer", total_layers=6, n=2):
    # freeze all
    for p in model.base_model.parameters():
        p.requires_grad = False
    # unfreeze last n
    start = total_layers - n
    for name, p in model.named_parameters():
        if any(name.startswith(f"{base_attr}.{i}") for i in range(start, total_layers)):
            p.requires_grad = True

def compute_weights(labels_np: np.ndarray, classes=(0,1,2)):
    w = compute_class_weight("balanced", classes=np.array(classes), y=labels_np)
    return torch.tensor(w, dtype=torch.float)

# -------- inference helper --------
def batch_predict_labels(texts, tok, model, device, max_len=MAX_LEN, bs=64):
    """Return label indices (no probs) for fast eval."""
    preds = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        inputs = tok(chunk, return_tensors="pt", truncation=True,
                     padding=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
    return preds
