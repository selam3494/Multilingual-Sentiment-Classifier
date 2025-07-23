# train_light_improved.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
)
from evaluate import load as load_metric

# ---------------- 0. Setup ----------------
model_name = "distilbert-base-multilingual-cased"
tok = AutoTokenizer.from_pretrained(model_name)
map3 = {0:0, 1:0, 2:1, 3:2, 4:2}

# ---------------- 1. Load + relabel ----------------
ds = load_dataset("mteb/amazon_reviews_multi", "en")
ds = ds.map(lambda x: {"labels": map3[x["label"]]})
ds = ds.rename_column("text", "sentence")
ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ["sentence", "labels"]])

# ---- Stratified sampling helper ----
def stratified_take(split, per_class, seed=42):
    idxs = []
    labels = np.array(split["labels"])
    rng = np.random.default_rng(seed)
    for c in np.unique(labels):
        cand = np.where(labels == c)[0]
        take = min(per_class, len(cand))
        idxs.extend(rng.choice(cand, take, replace=False))
    return split.select(sorted(idxs))

train_raw, val_raw, test_raw = ds["train"], ds["validation"], ds["test"]

train = stratified_take(train_raw.shuffle(seed=42), per_class=2000)   # 6000 total
val   = stratified_take(val_raw.shuffle(seed=42),   per_class=400)    # 1200 total
test  = stratified_take(test_raw.shuffle(seed=42),  per_class=600)    # 1800 total

print("Train:", Counter(train["labels"]))
print("Val  :", Counter(val["labels"]))
print("Test :", Counter(test["labels"]))

# ---------------- 2. Tokenize ----------------
def tok_batch(b):
    return tok(b["sentence"], truncation=True, padding=False, max_length=256)

train = train.map(tok_batch, batched=True)
val   = val.map(tok_batch, batched=True)
test  = test.map(tok_batch, batched=True)

cols = ["input_ids", "attention_mask", "labels"]
for split in (train, val, test):
    split.set_format(type="torch", columns=cols)

# ---------------- 3. Model ----------------
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Freeze all, unfreeze last 2 layers
for p in model.base_model.parameters():
    p.requires_grad = False
for name, p in model.named_parameters():
    if name.startswith("distilbert.transformer.layer.4") or name.startswith("distilbert.transformer.layer.5"):
        p.requires_grad = True

# ---------------- 4. Class weights ----------------
y_train = np.array(train["labels"])
weights = compute_class_weight(class_weight="balanced", classes=np.array([0,1,2]), y=y_train)
class_w = torch.tensor(weights, dtype=torch.float)

metric = load_metric("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"macro_f1": metric.compute(predictions=preds, references=labels, average="macro")["f1"]}

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = nn.CrossEntropyLoss(weight=class_w.to(outputs.logits.device))
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

args = TrainingArguments(
    output_dir="ckpt_light",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=1e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=2,
    report_to=[],
    no_cuda=True
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=val,
    tokenizer=tok,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
trainer.save_model("model_en_light_best")
tok.save_pretrained("model_en_light_best")
