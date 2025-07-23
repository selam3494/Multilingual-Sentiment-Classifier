# Multilingual-Sentiment-Classifier
**Short repo blurb (GitHub description field):**
Multilingual sentiment classifier: finetunes XLM-R on Spanish reviews, serves a tiny API, shows zero-shot vs finetuned F1.

---

**README opening section (copy-paste):**

# Multilingual Sentiment Classifier

A fast, no-drama project that proves you can wrangle transformers, transfer learning, and LLMs in under 48 hours.

**What it does**

* Finetunes a multilingual transformer (XLM-R or DistilBERT-multilingual) on Spanish Amazon reviews (amazon\_reviews\_multi).
* Outputs 3-way sentiment: negative, neutral, positive.
* Optional: compares against a zero-shot LLM baseline and can generate one-line rationales.

**Why it exists**
Because “I prompt ChatGPT” is not a portfolio. This shows actual model training, evaluation, and deployment.

**Stack**

* Hugging Face Transformers + PEFT (LoRA)
* Datasets library for loading and preprocessing
* FastAPI or Streamlit for a tiny UI
* Colab or local GPU for training

**Quick start**

```bash
pip install -r requirements.txt
python src/train.py        # finetune with LoRA on ~5k samples
python src/eval.py         # macro F1, confusion matrix
uvicorn src.app:app --reload   # serve /predict endpoint
```

**Results (example)**

| Model           | Macro F1 | Notes              |
| --------------- | -------- | ------------------ |
| Zero-shot LLM   | 0.xx     | 200-sample check   |
| Finetuned XLM-R | 0.yy     | LoRA r=8, 2 epochs |

**Next steps**

* Quantize with bitsandbytes
* Add active learning: let users correct labels and queue them for the next finetune
* Push model to Hugging Face Hub and deploy on Spaces

---

Need this trimmed, funnier, or tailored to another language? Say so.
