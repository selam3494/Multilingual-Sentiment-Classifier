# Multilingual Sentiment Classifier

<br>

| What you’ll do                                            | How long it takes |
| --------------------------------------------------------- | ----------------- |
| **Train & Fine‑tune** a multilingual DistilBERT on Amazon reviews | \~20 min on a T4  |
| **Evaluate** it (macro‑F1 & confusion matrix)             | seconds           |
| **Serve** it with FastAPI (locally **or** in Docker)      | 1 coffee          |
| **Predict** from the CLI or a REST call                   | milliseconds      |

---

## 🚀 1‑step install

```bash
git clone https://github.com/you/multilingual-sentiment-classifier.git
cd multilingual-sentiment-classifier
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # all libs below
```

<details>
<summary><code>requirements.txt</code> (click to view)</summary>

```text
# Core ML stack
torch==2.3.0
transformers==4.41.2
datasets==2.19.0
evaluate==0.4.2
scikit-learn==1.4.2
numpy>=1.24
matplotlib>=3.8
accelerate==0.30.1
tokenizers>=0.15.2
sentencepiece>=0.2.0
protobuf<5

# Serving
fastapi
uvicorn[standard]
```

</details>

---

## 🤖 Scripts – what they do

| File                    | Purpose                                                                                  |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `sentiment/trainer.py`  | Fine‑tunes the model (LoRA‑friendly, class‑weighting, frozen base except last 2 layers). |
| `sentiment/evaluate.py` | Loads a saved model, prints macro‑F1 & saves `confusion_matrix.png`.                     |
| `sentiment/model_io.py` | Tiny helpers: load model, predict one, predict batch.                                    |
| `sentiment/cli.py`      | Swiss‑army knife: `train`, `eval`, `predict` from the command line.                      |
| `api/main.py`           | FastAPI server with `/health`, `/predict`, `/predict_batch`.                             |
| `Dockerfile`            | Slim Python 3.10 image for serving (CPU‑only).                                           |
| `docker-compose.yml`    | Mounts the trained model & publishes port 8000.                                          |

The rest (`constants.py`, `data_utils.py`, `model_utils.py`) are just imported helpers—keep ’em.

---

## 🏋️‍♂️ Train / Evaluate / Predict (host machine)

```bash
# Train (creates model_en_light_best/)
python -m sentiment.cli train --lang en

# Evaluate
python -m sentiment.cli eval --model_dir model_en_light_best --lang en

# Predict from CLI
python -m sentiment.cli predict "I love this product!" --model_dir model_en_light_best
```

---

## 🌐 Serve with FastAPI (no Docker)

```bash
uvicorn api.main:app --reload \
       --port 8000 \
       --env-file <(echo MODEL_DIR=model_en_light_best)
```

* GET  **/health** → `{ "status": "ok" }`
* POST **/predict**  `{ "text": "c'est nul" }` → sentiment JSON

---

## 🐳 Serve **inside Docker**

1. **Build image**

```bash
docker compose build          # or: docker build -t sentiment-api .
```

2. **Run** (mounts model, maps port 8000):

```bash
docker compose up             # CTRL‑C to stop
```

<details>
<summary>docker-compose.yml (already in repo)</summary>

```yaml
version: "3.9"
services:
  sentiment-api:
    build: .
    container_name: sentiment-api
    ports:
      - "8000:8000"
    environment:
      MODEL_DIR: /app/model_en_light_best
      CONF_THRESH: "0.6"
    volumes:
      - ./model_en_light_best:/app/model_en_light_best:ro
```

</details>

3. **Hit the endpoint**

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"Esto es fantástico"}'
```

---

## 📝 Typical results (EN model, 5 k train)

| Model                          | Macro F1 |
| ------------------------------ | -------- |
| DistilBERT‑multilingual (ours) | **0.87** |
| GPT‑3.5 zero‑shot              | 0.72     |

*(Numbers vary ±0.02)*

---

## 🛠️ Next ideas

* Quantize with bits‑and‑bytes → 4 × faster inference
* Add Streamlit front‑end
* Push to Hugging Face Hub & deploy on Spaces
* Plug active‑learning loop so users can relabel mistakes live

---

### FAQ

*“Where should I place `model_en_light_best/`?”*
At repo root (same level as `Dockerfile`). The compose file mounts it into the container.

*“Can I train inside Docker?”*
Sure, but this image is CPU‑only. Use the existing scripts on a GPU machine instead.
