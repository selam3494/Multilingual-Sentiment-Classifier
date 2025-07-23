# api/main.py
import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from sentiment.model_io import load_model, predict_one, predict_batch

MODEL_DIR = os.getenv("MODEL_DIR", "model_en_light_best")
CONF_THRESH = float(os.getenv("CONF_THRESH", "0.6"))

app = FastAPI(title="Sentiment API", version="1.0.0")

tok, model, device = load_model(MODEL_DIR)

class TextIn(BaseModel):
    text: str
    thresh: Optional[float] = None

class BatchIn(BaseModel):
    texts: List[str]
    thresh: Optional[float] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_endpoint(body: TextIn):
    thresh = body.thresh if body.thresh is not None else CONF_THRESH
    return predict_one(body.text, tok, model, device, thresh)

@app.post("/predict_batch")
def predict_batch_endpoint(body: BatchIn):
    thresh = body.thresh if body.thresh is not None else CONF_THRESH
    return {"results": predict_batch(body.texts, tok, model, device, thresh)}

@app.get("/predict")
def predict_query(text: str = Query(...), thresh: float = CONF_THRESH):
    return predict_one(text, tok, model, device, thresh)
