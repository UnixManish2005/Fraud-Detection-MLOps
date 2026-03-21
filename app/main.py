"""
app/main.py
-----------
FastAPI application exposing the fraud detection model as a REST API.

Endpoints
---------
GET  /              — health check
GET  /health        — detailed health check with model info
POST /predict       — single transaction prediction
POST /predict/batch — batch transaction predictions

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# ── Path fix ───────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.predict import predictor

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "api.log")),
    ],
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading fraud detection model …")
    try:
        predictor.load()
        logger.info("Model ready: %s", predictor.model_name)
    except FileNotFoundError as e:
        logger.error("Model not found: %s", e)
        logger.error("Train the model first: python src/model_training.py --data data/creditcard.csv")
    yield
    logger.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time credit card fraud detection using ML.\n\n"
        "Train a model with `python src/model_training.py` then hit POST /predict."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class TransactionFeatures(BaseModel):
    """
    Raw transaction features.  V1-V28 are PCA-transformed by the issuer.
    Amount is in the original currency.  Time (seconds since first transaction
    in the dataset) is optional — a zero placeholder is used if absent.
    """
    Time: Optional[float] = Field(None, description="Seconds since first transaction (optional)")
    V1:  float; V2:  float; V3:  float; V4:  float
    V5:  float; V6:  float; V7:  float; V8:  float
    V9:  float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount (>= 0)")

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 406,
                "V1": -1.3598071, "V2": -0.0727812, "V3":  2.5363467,
                "V4":  1.3781553, "V5": -0.3383208, "V6":  0.4623878,
                "V7":  0.2395986, "V8":  0.0986980, "V9":  0.3637870,
                "V10": 0.0907942, "V11": -0.5515995, "V12": -0.6178009,
                "V13": -0.9913898, "V14": -0.3111694, "V15":  1.4681770,
                "V16": -0.4704005, "V17":  0.2079708, "V18":  0.0257906,
                "V19":  0.4039936, "V20":  0.2514121, "V21": -0.0183067,
                "V22":  0.2778376, "V23": -0.1104739, "V24":  0.0669281,
                "V25":  0.1285394, "V26": -0.1891148, "V27":  0.1335584,
                "V28": -0.0210530,
                "Amount": 149.62,
            }
        }


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    model_name: str
    threshold: float
    latency_ms: float


class BatchRequest(BaseModel):
    transactions: list[TransactionFeatures]

    @validator("transactions")
    def check_not_empty(cls, v):
        if not v:
            raise ValueError("transactions list must not be empty")
        if len(v) > 1000:
            raise ValueError("Maximum batch size is 1000")
        return v


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    fraud_count: int


# ── Request timing middleware ──────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    return response


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "Fraud Detection API is running."}


@app.get("/health", tags=["Health"])
async def health():
    loaded = predictor._loaded
    return {
        "status": "healthy" if loaded else "degraded",
        "model_loaded": loaded,
        "model_name": predictor.model_name if loaded else None,
        "threshold": predictor.threshold if loaded else None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(transaction: TransactionFeatures):
    """
    Predict whether a single credit card transaction is fraudulent.

    **Returns**
    - `fraud_probability` — model confidence (0–1) that this is fraud
    - `is_fraud` — True if probability exceeds the decision threshold
    - `model_name` — name of the underlying model
    - `threshold` — decision threshold in use
    - `latency_ms` — server-side inference latency
    """
    if not predictor._loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")

    t0 = time.perf_counter()
    raw = transaction.dict()
    try:
        result = predictor.predict(raw)
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "predict | fraud_prob=%.4f | is_fraud=%s | latency=%.1f ms",
        result["fraud_probability"], result["is_fraud"], latency_ms,
    )
    return PredictionResponse(**result, latency_ms=round(latency_ms, 2))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
async def predict_batch(body: BatchRequest):
    """
    Predict fraud for a batch of up to 1 000 transactions in one call.
    """
    if not predictor._loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")

    records = [t.dict() for t in body.transactions]
    try:
        results = predictor.predict_batch(records)
    except Exception as e:
        logger.exception("Batch prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

    predictions = [
        PredictionResponse(**r, latency_ms=0.0) for r in results
    ]
    fraud_count = sum(1 for p in predictions if p.is_fraud)
    logger.info("predict/batch | total=%d | fraud_count=%d", len(predictions), fraud_count)

    return BatchResponse(
        predictions=predictions,
        total=len(predictions),
        fraud_count=fraud_count,
    )


# ── Exception handler ──────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
