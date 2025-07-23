# sentiment/trainer.py
import torch, numpy as np, torch.nn as nn
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding,
                          EarlyStoppingCallback)
from evaluate import load as load_metric
from .constants import MAP3, MAX_LEN
from .data_utils import load_amazon_en, stratified_take

def tok_batch(tok):
    def _fn(b):
        return tok(b["sentence"], truncation=True, padding=False, max_length=MAX_LEN)
    return _fn

def train_model(output_dir="model_en_light_best", model_name="distilbert-base-multilingual-cased"):
    tok = AutoTokenizer.from_pretrained(model_name)

    ds = load_amazon_en(MAP3)
    ds = ds.rename_column("text", "sentence")
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ["sentence","labels"]])

    train_raw, val_raw, test_raw = ds["train"], ds["validation"], ds["test"]
    train = stratified_take(train_raw.shuffle(seed=42), 2000)
    val   = stratified_take(val_raw.shuffle(seed=42),   400)
    test  = stratified_take(test_raw.shuffle(seed=42),  600)

    print("Train:", Counter(train["labels"]))
    print("Val  :", Counter(val["labels"]))
    print("Test :", Counter(test["labels"]))

    train = train.map(tok_batch(tok), batched=True)
    val   = val.map(tok_batch(tok), batched=True)
    test  = test.map(tok_batch(tok), batched=True)

    cols = ["input_ids","attention_mask","labels"]
    for split in (train, val, test):
        split.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    for p in model.base_model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith("distilbert.transformer.layer.4") or name.startswith("distilbert.transformer.layer.5"):
            p.requires_grad = True

    y_train = np.array(train["labels"])
    weights = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y_train)
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
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
