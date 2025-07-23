# sentiment/trainer.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_TF"] = "1"   # stop TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # silence TF if it still sneaks in

from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
)
from evaluate import load as load_metric

from .constants import (
    MAP5_TO_3, MAX_LEN, NUM_LABELS,
    DEFAULT_MODEL, EPOCHS, BS_TRAIN, BS_EVAL, LR, SEED
)
from .data_utils import load_amazon, stratified_take

def _tok_batch(tok):
    def _fn(b):
        return tok(b["sentence"], truncation=True, padding=False, max_length=MAX_LEN)
    return _fn

def train_model(
    lang: str = "en",
    model_name: str = DEFAULT_MODEL,
    per_class_train: int = 2000,
    per_class_val: int = 400,
    per_class_test: int = 600,
    output_dir: str = "model_en_light_best",
    use_cuda: bool = True
):
    # 1) Load & relabel
    ds = load_amazon(lang, MAP5_TO_3)
    ds = ds.rename_column("text", "sentence")
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ["sentence", "labels"]])

    train_raw, val_raw, test_raw = ds["train"], ds["validation"], ds["test"]
    train = stratified_take(train_raw.shuffle(seed=SEED), per_class_train)
    val   = stratified_take(val_raw.shuffle(seed=SEED),  per_class_val)
    test  = stratified_take(test_raw.shuffle(seed=SEED), per_class_test)

    print("Train:", Counter(train["labels"]))
    print("Val  :", Counter(val["labels"]))
    print("Test :", Counter(test["labels"]))

    # 2) Tokenize
    tok = AutoTokenizer.from_pretrained(model_name)
    train = train.map(_tok_batch(tok), batched=True)
    val   = val.map(_tok_batch(tok),   batched=True)
    test  = test.map(_tok_batch(tok),  batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    for split in (train, val, test):
        split.set_format(type="torch", columns=cols)

    # 3) Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)

    # Freeze all, unfreeze last 2
    for p in model.base_model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith("distilbert.transformer.layer.4") or name.startswith("distilbert.transformer.layer.5"):
            p.requires_grad = True

    # 4) Class weights
    y_train = np.array(train["labels"])
    weights = compute_class_weight("balanced", classes=np.array(range(NUM_LABELS)), y=y_train)
    class_w = torch.tensor(weights, dtype=torch.float)

    # 5) Metrics
    metric = load_metric("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        f1 = metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        return {"macro_f1": f1}

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fct = nn.CrossEntropyLoss(weight=class_w.to(outputs.logits.device))
            loss = loss_fct(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir="ckpt_light",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BS_TRAIN,
        per_device_eval_batch_size=BS_EVAL,
        learning_rate=LR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to=[],
        seed=SEED,
        no_cuda=not use_cuda
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
    best_metric = float(trainer.state.best_metric)

    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    return output_dir, best_metric
