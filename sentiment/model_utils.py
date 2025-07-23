import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .constants import MAX_LEN

def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tok, model, device

def batch_predict(texts, tok, model, device, max_len=MAX_LEN, bs=64):
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
