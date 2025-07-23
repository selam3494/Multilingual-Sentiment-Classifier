# api/main.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

from sentiment.model_io import load_model, predict_one, predict_batch

MODEL_DIR = os.getenv("MODEL_DIR", "model_en_light_best")
CONF_THRESH = float(os.getenv("CONF_THRESH", "0.6"))

app = FastAPI(title="Sentiment API", version="1.0.0")

class TextIn(BaseModel):
    text: str
    thresh: Optional[float] = None

class BatchIn(BaseModel):
    texts: List[str]
    thresh: Optional[float] = None

@app.on_event("startup")
async def _startup():
    p = Path(MODEL_DIR)
    if not p.exists():
        raise RuntimeError(f"MODEL_DIR '{MODEL_DIR}' not found. Train first or set MODEL_DIR to a valid path.")
    tok, model, device = load_model(str(p))
    app.state.tok = tok
    app.state.model = model
    app.state.device = device

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_endpoint(body: TextIn):
    thresh = body.thresh if body.thresh is not None else CONF_THRESH
    return predict_one(body.text, app.state.tok, app.state.model, app.state.device, thresh)

@app.post("/predict_batch")
def predict_batch_endpoint(body: BatchIn):
    thresh = body.thresh if body.thresh is not None else CONF_THRESH
    return {"results": predict_batch(body.texts, app.state.tok, app.state.model, app.state.device, thresh)}

@app.get("/predict")
def predict_query(text: str = Query(...), thresh: float = CONF_THRESH):
    return predict_one(text, app.state.tok, app.state.model, app.state.device, thresh)
