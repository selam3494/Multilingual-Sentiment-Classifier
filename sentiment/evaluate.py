# sentiment/evaluate.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from datasets import load_dataset

from .constants import LABELS, MAP5_TO_3
from .model_io import load_model
from .model_utils import batch_predict_labels

def eval_model(model_dir="model_en_light_best", lang="en", n_test=None, cm_path="confusion_matrix.png"):
    tok, model, device = load_model(model_dir)

    ds = load_dataset("mteb/amazon_reviews_multi", lang)
    ds = ds.map(lambda x: {"labels": MAP5_TO_3[x["label"]]})
    test = ds["test"] if n_test is None else ds["test"].select(range(n_test))
    texts = test["text"] if "text" in test.column_names else test["sentence"]
    y_true = test["labels"]

    y_pred = batch_predict_labels(texts, tok, model, device)

    macro = f1_score(y_true, y_pred, average="macro")
    print("Macro F1:", macro)
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

    return macro, cm_path
