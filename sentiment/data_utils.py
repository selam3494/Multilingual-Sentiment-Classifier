# sentiment/data_utils.py
import numpy as np
from datasets import load_dataset

def load_amazon(lang: str, map5_to_3: dict):
    """
    lang: 'en' or 'fr' (extend if you add more)
    """
    ds = load_dataset("mteb/amazon_reviews_multi", lang)
    ds = ds.map(lambda x: {"labels": map5_to_3[x["label"]]})
    return ds

def stratified_take(split, per_class, seed=42):
    idxs = []
    labels = np.array(split["labels"])
    rng = np.random.default_rng(seed)
    for c in np.unique(labels):
        cand = np.where(labels == c)[0]
        take = min(per_class, len(cand))
        idxs.extend(rng.choice(cand, take, replace=False))
    return split.select(sorted(idxs))
