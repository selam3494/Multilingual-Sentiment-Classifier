import numpy as np
from datasets import load_dataset
from .constants import MAP5_TO_3, SEED

def load_and_prepare(lang="en"):
    ds = load_dataset("mteb/amazon_reviews_multi", lang)
    ds = ds.map(lambda x: {"labels": MAP5_TO_3[x["label"]]})
    ds = ds.rename_column("text", "sentence")
    keep_cols = ["sentence", "labels"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep_cols])
    return ds

def stratified_take(split, per_class, seed=SEED):
    idxs = []
    labels = np.array(split["labels"])
    rng = np.random.default_rng(seed)
    for c in np.unique(labels):
        cand = np.where(labels == c)[0]
        take = min(per_class, len(cand))
        idxs.extend(rng.choice(cand, take, replace=False))
    return split.select(sorted(idxs))
