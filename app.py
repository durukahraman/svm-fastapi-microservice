from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, joblib
from pathlib import Path

app = FastAPI(title="SVM Microservice", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

MODEL_FILE = Path(__file__).parent / "model.joblib"
if not MODEL_FILE.exists():
    raise RuntimeError(f"Model bulunamadı: {MODEL_FILE}")

model = joblib.load(MODEL_FILE)
N_FEATURES = getattr(model, "n_features_in_", None)

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok", "n_features": int(N_FEATURES or 0)}

@app.post("/predict")
def predict(req: PredictRequest):
    x = np.array(req.features, dtype=float).reshape(1, -1)
    if N_FEATURES is not None and x.shape[1] != N_FEATURES:
        raise HTTPException(status_code=400, detail=f"{N_FEATURES} özellik bekleniyor.")
    pred = int(model.predict(x)[0])
    proba = float(model.predict_proba(x)[0].max()) if hasattr(model, "predict_proba") else None
    return {"prediction": pred, "class": pred, "proba": proba}

