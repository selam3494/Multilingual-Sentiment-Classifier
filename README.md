Below is a **drop‑in replacement** for your README that

1. Adds an **“Experiment tracking with MLflow”** section (how to launch the UI, where runs are saved, sample screenshot table).
2. Includes the three new screenshots (`Screenshot from 2025‑07‑25 15‑34‑27.png`, `…41.png`, `…52.png`) alongside the existing ones.

Copy‑paste over the old `README.md`, commit, and push — GitHub will render the images as long as the `.png` files live in the same folder (spaces already encoded as `%20`).

````markdown
## Multilingual Sentiment Classifier

A project that proves you can wrangle Transformers, transfer‑learning & deploy it with FastAPI.  
It trains a **3‑class** sentiment model (negative / neutral / positive) on multilingual Amazon reviews, evaluates it, logs everything to **MLflow**, and serves predictions through a lightweight API—plus Docker for one‑command deployment.

---

## 1‑step install

```bash
git clone https://github.com/you/multilingual-sentiment-classifier.git
cd multilingual-sentiment-classifier
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
````

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

# Tracking & serving
mlflow>=2.12
fastapi
uvicorn[standard]
```

</details>

---

## Scripts – what they do

| File                    | Purpose                                                                                                            |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `sentiment/trainer.py`  | Fine‑tunes the model (class‑weights, freezes base except last 2 layers) **and logs parameters/metrics to MLflow.** |
| `sentiment/evaluate.py` | Loads a saved model, prints macro‑F1, saves `confusion_matrix.png`, **logs to MLflow**.                            |
| `sentiment/model_io.py` | Helpers: load model, predict one/batch.                                                                            |
| `sentiment/cli.py`      | Swiss‑army knife: `train`, `eval`, `predict` commands.                                                             |
| `api/main.py`           | FastAPI server with `/health`, `/predict`, `/predict_batch`.                                                       |
| `Dockerfile`            | Slim Python 3.10 image for serving (CPU‑only).                                                                     |
| `docker-compose.yml`    | Mounts the trained model & publishes port 8000.                                                                    |
| Other helpers           | `constants.py`, `data_utils.py`, `model_utils.py`—imported utilities.                                              |

---

## Train / Evaluate / Predict (host machine)

```bash
# Train (creates model_en_light_best/ + MLflow run)
python -m sentiment.cli train --lang en

# Evaluate (logs metric + confusion matrix in the same experiment)
python -m sentiment.cli eval --model_dir model_en_light_best --lang en

# Predict from CLI
python -m sentiment.cli predict "I love this product!" --model_dir model_en_light_best
```

---

## Experiment tracking with MLflow

```bash
# Launch the tracking UI in another terminal/tab
mlflow ui --backend-store-uri mlruns --port 5000
# → http://localhost:5000
```

Every `train`/`eval` command creates a run inside the *Sentiment\_Evaluation* experiment:

| Screenshot                                                    | What you see                               |
| ------------------------------------------------------------- | ------------------------------------------ |

| ![Mlflow](./Screenshot%20from%202025-07-25%2015-34-52.png) |
| ![Experiment](./Screenshot%20from%202025-07-25%2015-34-27.png)  | All runs with `macro_f1`, lang, model hash |
| ![Mlflow Parameters](./Screenshot%20from%202025-07-25%2015-34-41.png) |

The default `backend-store-uri` is the local `mlruns/` folder—commit it or point to an S3/DB URI for team sharing.

---

## Serve with FastAPI (no Docker)

```bash
uvicorn api.main:app --reload \
       --port 8000 \
       --env-file <(echo MODEL_DIR=model_en_light_best)
```

* **GET /health** → `{"status":"ok"}`
* **POST /predict**  `{"text":"c'est nul"}` → sentiment JSON

---

## Serve **inside Docker**

1. **Build**

```bash
docker compose build          # or: docker build -t sentiment-api .
```

2. **Run**

```bash
docker compose up
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

## Quick tour (click to zoom)

| Step                           | Screenshot                                                    |
| ------------------------------ | ------------------------------------------------------------- |
| **Train/Predict/Evaluate CLI** | ![CLI](./Screenshot%20from%202025-07-24%2010-49-21.png)       |
| **Training loop**              | ![Training](./Screenshot%20from%202025-07-24%2010-48-26.png)  |
| **Evaluation matrix**          | ![Confusion](./Screenshot%20from%202025-07-24%2010-48-55.png) |

---

## Next ideas

* Quantize with bits‑and‑bytes → ↘ latency / ↘ RAM
* Streamlit front‑end
* Push model to Hugging Face Hub & launch on Spaces
* Active‑learning loop so users can correct predictions live
* Try a GPU‑heavier backbone (XLM‑R large) for +F1

---

### FAQ

**Where should I place `model_en_light_best/`?**
At repo root (same level as `Dockerfile`). `docker‑compose.yml` mounts it for the container.

**Can I train inside Docker?**
Yes, but the provided image is CPU‑only. Train on a GPU box, then copy the exported folder back.

```

*That’s it—README now documents MLflow usage and shows the new screenshots.*
::contentReference[oaicite:0]{index=0}
```
