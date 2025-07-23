import torch, numpy as np, matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report, confusion_matrix

LABELS = ["negative","neutral","positive"]
MAP3 = {0:0, 1:0, 2:1, 3:2, 4:2}

def load_model(path="model_fr_light_best"):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tok, model, device

def batch_predict(texts, tok, model, device, max_len=256, bs=64):
    preds = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        inputs = tok(chunk, return_tensors="pt", truncation=True,
                     padding=True, max_length=max_len)
        inputs = {k: v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
    return preds

def eval_model(model_dir="model_fr_light_best", n_test=None, cm_path="confusion_matrix.png"):
    tok, model, device = load_model(model_dir)
    ds = load_dataset("mteb/amazon_reviews_multi", "fr")
    ds = ds.map(lambda x: {"labels": MAP3[x["label"]]})
    test = ds["test"] if n_test is None else ds["test"].select(range(n_test))
    texts = test["text"] if "text" in test.column_names else test["sentence"]
    y_true = test["labels"]
    y_pred = batch_predict(texts, tok, model, device)

    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred, target_names=LABELS))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS))); ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")