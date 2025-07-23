import torch
from .model_utils import load_model
from .constants import LABELS, MAX_LEN

def predict_one(text, model_dir="model_en_light_best", thresh=0.6):
    tok, model, device = load_model(model_dir)
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1).squeeze().tolist()

    probs_dict = {lab: round(p * 100, 2) for lab, p in zip(LABELS, probs)}
    pred = max(probs_dict, key=probs_dict.get)
    confident = probs_dict[pred] >= thresh * 100

    return {
        "text": text,
        "prediction": pred,
        "confidence_%": probs_dict[pred],
        "all_probs_%": probs_dict,
        "confident": confident
    }

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("text", help="Text to classify")
    ap.add_argument("--model_dir", default="model_en_light_best")
    ap.add_argument("--thresh", type=float, default=0.6)
    args = ap.parse_args()

    out = predict_one(args.text, args.model_dir, args.thresh)
    print(json.dumps(out, indent=2, ensure_ascii=False))
