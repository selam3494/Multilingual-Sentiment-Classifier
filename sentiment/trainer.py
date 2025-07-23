import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
)
from evaluate import load as load_metric

from .constants import (
    DEFAULT_MODEL, MAX_LEN, EPOCHS, BS_TRAIN, BS_EVAL, LR, SEED
)
from .data_utils import load_and_prepare, stratified_take

def tokenize_dataset(ds, tok):
    def tok_batch(b):
        return tok(b["sentence"], truncation=True, padding=False, max_length=MAX_LEN)
    return ds.map(tok_batch, batched=True)

def freeze_except_last_two(model):
    for p in model.base_model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith("distilbert.transformer.layer.4") or name.startswith("distilbert.transformer.layer.5"):
            p.requires_grad = True

def train_model(
    lang="en",
    model_name=DEFAULT_MODEL,
    per_class_train=2000,
    per_class_val=400,
    per_class_test=600,
    output_dir="ckpt_light",
    use_cuda=True
):
    # 1. Data
    ds = load_and_prepare(lang)
    train_raw, val_raw, test_raw = ds["train"], ds["validation"], ds["test"]

    train = stratified_take(train_raw.shuffle(seed=SEED), per_class=per_class_train)
    val   = stratified_take(val_raw.shuffle(seed=SEED),   per_class=per_class_val)
    test  = stratified_take(test_raw.shuffle(seed=SEED),  per_class=per_class_test)

    tok = AutoTokenizer.from_pretrained(model_name)
    train = tokenize_dataset(train, tok)
    val   = tokenize_dataset(val, tok)
    test  = tokenize_dataset(test, tok)

    cols = ["input_ids", "attention_mask", "labels"]
    for split in (train, val, test):
        split.set_format(type="torch", columns=cols)

    # 2. Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    freeze_except_last_two(model)

    # 3. Class weights
    y_train = np.array(train["labels"])
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0,1,2]), y=y_train)
    class_w = torch.tensor(weights, dtype=torch.float)

    # 4. Metrics
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
        output_dir=output_dir,
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
        no_cuda=not use_cuda,
        seed=SEED
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

    best_dir = f"{output_dir}/{args.run_name or ''}".strip("/")
    save_dir = "model_{}_light_best".format(lang)
    trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)

    return save_dir, trainer.state.best_metric
