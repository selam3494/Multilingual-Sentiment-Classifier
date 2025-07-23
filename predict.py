LABELS = ["negative", "neutral", "positive"]

def predict_one(text, model_dir="model_en_light_best", thresh=0.6):
    tok, model, device = load_model(model_dir)
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1).squeeze().tolist()

    probs_dict = {lab: round(p*100, 2) for lab, p in zip(LABELS, probs)}
    pred = max(probs_dict, key=probs_dict.get)

    # optional: flag low-confidence cases
    confident = probs_dict[pred] >= thresh*100

    return {
        "text": text,
        "prediction": pred,
        "confidence_%": probs_dict[pred],
        "all_probs_%": probs_dict,
        "confident": confident
    }