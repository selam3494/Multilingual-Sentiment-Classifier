## Multilingual Sentiment Classifier

A project that proves you can wrangle Transformers, transfer‑learning & FastAPI in under 48 hours.
It trains a 3‑class sentiment model (negative / neutral / positive) on multilingual Amazon reviews, evaluates it, and serves predictions through a lightweight API—plus Docker for one‑command deployment.

## 1‑step install

```bash
git clone https://github.com/you/multilingual-sentiment-classifier.git
cd multilingual-sentiment-classifier
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
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

GitHub renders README markdown natively; images referenced with relative paths resolve automatically after commit ([GitHub Docs][3]).

---

## Scripts – what they do

| File                    | Purpose                                                                  |
| ----------------------- | ------------------------------------------------------------------------ |
| `sentiment/trainer.py`  | Fine‑tunes the model (class‑weights, freezes base except last 2 layers). |
| `sentiment/evaluate.py` | Loads a saved model, prints macro‑F1 & writes `confusion_matrix.png`.    |
| `sentiment/model_io.py` | Helpers: load model, predict one/batch.                                  |
| `sentiment/cli.py`      | Swiss‑army knife: `train`, `eval`, `predict` commands.                   |
| `api/main.py`           | FastAPI server with `/health`, `/predict`, `/predict_batch`.             |
| `Dockerfile`            | Slim Python 3.10 image for serving (CPU‑only).                           |
| `docker-compose.yml`    | Mounts the trained model & publishes port 8000.                          |
| Other helpers           | `constants.py`, `data_utils.py`, `model_utils.py`—imported utilities.    |

---

## Train / Evaluate / Predict (host machine)

```bash
# Train (creates model_en_light_best/)
python -m sentiment.cli train --lang en

# Evaluate
python -m sentiment.cli eval --model_dir model_en_light_best --lang en

# Predict from CLI
python -m sentiment.cli predict "I love this product!" --model_dir model_en_light_best
```

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
docker compose build          # or docker build -t sentiment-api .
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

### Quick tour (click to zoom)

| Step                | Screenshot                                                                     |
| ------------------- | ------------------------------------------------------------------------------ |
| **1 – Training**    | ![Training log](./Screenshot%20from%202025-07-24%2010-48-26.png)               |
| **2 – CLI predict** | ![Single‑text inference](./Screenshot%20from%202025-07-24%2010-49-21.png)      |
| **3 – Evaluation**  | ![Confusion matrix + metrics](./Screenshot%20from%202025-07-24%2010-48-55.png) |


*(If GitHub preview doesn’t show the images, check that the files exist in the same folder or replace spaces with `%20` as above) ([Stack Overflow][1], [Stack Overflow][2])*

---

## Next ideas

* Quantize with bits‑and‑bytes → 4 × faster inference
* Streamlit front‑end
* Push model to Hugging Face Hub & launch on Spaces
* Active‑learning loop so users can correct predictions live

---

### FAQ

**Where should I place `model_en_light_best/`?**
At repo root (same level as `Dockerfile`). `docker‑compose.yml` mounts it inside the container.

**Can I train inside Docker?**
Yes, but the provided image is CPU‑only. Train on a GPU machine for speed, then copy the exported folder back.

---

[1]: https://stackoverflow.com/questions/14494747/how-to-add-images-to-readme-md-on-github?utm_source=chatgpt.com "How to add images to README.md on GitHub? - Stack Overflow"
[2]: https://stackoverflow.com/questions/15764242/is-it-possible-to-make-relative-link-to-image-in-a-markdown-file-in-a-gist?utm_source=chatgpt.com "Is it possible to make relative link to image in a markdown file in a gist?"
[3]: https://docs.github.com/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes?utm_source=chatgpt.com "About READMEs - GitHub Docs"
