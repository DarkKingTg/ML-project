"""
api/app.py

FastAPI HTTP layer exposing the BehaveGuard prediction service.

Endpoints:
    GET  /health          -- liveness/readiness check
    POST /predict          -- classify a single prompt
    POST /predict/batch    -- classify multiple prompts

Run with:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.inference_service import InferenceService
from config.settings import load_config
from utils.logger import get_logger
from utils.exceptions import PredictionError, ModelLoadError, ConfigError

logger = get_logger(__name__)

CONFIG_PATH = "config/config.yaml"

# Populated at startup via the lifespan handler below, rather than at import
# time, so import errors during artifact loading surface as a clear startup
# failure rather than a confusing import-time crash.
_service_holder: dict = {"service": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the InferenceService once when the API process starts."""
    try:
        _service_holder["service"] = InferenceService(CONFIG_PATH)
        logger.info("InferenceService loaded successfully at API startup")
    except (ModelLoadError, ConfigError) as e:
        logger.error(f"Failed to initialize InferenceService at startup: {e}")
        # Re-raise so the server fails to start rather than serving requests
        # against a half-initialized service.
        raise
    yield
    logger.info("API shutting down")


def get_service() -> InferenceService:
    """
    Retrieve the loaded InferenceService, raising a clear HTTP error if the
    API somehow received a request before startup completed.
    """
    service = _service_holder["service"]
    if service is None:
        raise HTTPException(status_code=503, detail="Service not yet initialized")
    return service


try:
    _api_cfg = load_config(CONFIG_PATH)["api"]
except ConfigError:
    _api_cfg = {"title": "BehaveGuard Prompt Injection Detection API"}

app = FastAPI(title=_api_cfg.get("title", "BehaveGuard API"), lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #

class PredictRequest(BaseModel):
    """Request body for a single prediction."""
    text: str = Field(..., min_length=1, description="The prompt text to classify")


class BatchPredictRequest(BaseModel):
    """Request body for batch prediction."""
    texts: list[str] = Field(..., min_length=1, description="List of prompt texts to classify")


class PredictResponse(BaseModel):
    """Response body for a prediction result."""
    prompt: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health")
def health() -> dict:
    """
    Liveness/readiness check. Returns 200 with status 'ok' once the model is
    loaded, or a 503 (via get_service) if called before startup completes.
    """
    service = get_service()
    return {
        "status": "ok",
        "embedder_backend": type(service.embedder).__name__,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    """
    Classify a single prompt as Safe or Malicious.

    Returns:
        Prediction with confidence and per-class probabilities.

    Raises:
        HTTPException 400: If the input is invalid (e.g. empty after validation).
        HTTPException 500: If prediction fails unexpectedly.
    """
    service = get_service()
    try:
        return service.predict(request.text)
    except PredictionError as e:
        logger.warning(f"Prediction request failed validation/processing: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(request: BatchPredictRequest) -> list:
    """
    Classify multiple prompts in a single request.

    Returns:
        List of predictions, same order as the input texts.

    Raises:
        HTTPException 400: If any input is invalid.
        HTTPException 500: If prediction fails unexpectedly.
    """
    service = get_service()
    try:
        return service.predict_batch(request.texts)
    except PredictionError as e:
        logger.warning(f"Batch prediction request failed validation/processing: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")
