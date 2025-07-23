# sentiment/data_utils.py
import numpy as np
from datasets import load_dataset

def load_amazon_fr(map3):
    ds = load_dataset("mteb/amazon_reviews_multi", "fr")
    ds = ds.map(lambda x: {"labels": map3[x["label"]]})
    return ds

def load_amazon_en(map3):
    ds = load_dataset("mteb/amazon_reviews_multi", "en")
    ds = ds.map(lambda x: {"labels": map3[x["label"]]})
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
