from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path

from .schemas import TransactionRequest, PredictionResponse, HealthResponse
from .predictor import FraudPredictor

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

predictor: FraudPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = FraudPredictor()
    print(f"Model loaded: {predictor.model_name}  threshold={predictor.threshold}")
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="XGBoost-based real-time fraud scoring for the IBM synthetic dataset.",
    version="1.0.0",
    lifespan=lifespan,
)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model=predictor.model_name,
        threshold=predictor.threshold,
        features=len(predictor.feature_order),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: TransactionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = predictor.predict(request)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
