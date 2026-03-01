import logging
import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Path bootstrap
# main.py lives at nvda_predictor/api/ — walk up to nvda_predictor/ then into src/
# ---------------------------------------------------------------------------
nvda_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../nvda_predictor/
src_root  = os.path.join(nvda_root, "src")                               # .../nvda_predictor/src/
if src_root not in sys.path:
    sys.path.insert(0, src_root)

# Load .env from nvda_predictor/ — must happen before any os.getenv() call
load_dotenv(os.path.join(nvda_root, ".env"))

from model.predict import predict_next_hour, _load_model_and_scaler  # noqa: E402

# ---------------------------------------------------------------------------
# Logging — must be configured before any logger.info() call
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CORS origins — read from .env after load_dotenv() has run
# ---------------------------------------------------------------------------
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]
logger.info("CORS origins: %s", ALLOWED_ORIGINS)


# ---------------------------------------------------------------------------
# Lifespan — model loaded once at startup, released at shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading LSTM model and scaler into memory...")
    try:
        model, scaler_min, scaler_max = _load_model_and_scaler()
        app.state.model      = model
        app.state.scaler_min = scaler_min
        app.state.scaler_max = scaler_max
        logger.info("Model and scaler ready. Server accepting requests.")
    except FileNotFoundError as exc:
        logger.critical("STARTUP FAILED — asset not found: %s", exc)
        raise

    yield

    logger.info("Shutdown: releasing model assets.")
    app.state.model      = None
    app.state.scaler_min = None
    app.state.scaler_max = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NVDA AI Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def read_root():
    """Basic liveness check — confirms the server process is running."""
    return {"status": "AI Server is Online", "model": "NVDA-LSTM"}


@app.get("/health", tags=["meta"])
def health_check():
    """
    Health endpoint for load balancers and uptime monitors.
    Returns 200 when the model is loaded and ready, 503 otherwise.
    """
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.get("/predict", tags=["prediction"])
def get_prediction():
    """
    Run NVDA next-hour price prediction.

    Intentionally a plain `def` (not `async def`) — FastAPI runs sync
    endpoints in a thread-pool executor, which is correct here because
    predict_next_hour() performs blocking I/O and CPU-bound inference.
    """
    try:
        result = predict_next_hour(
            model=app.state.model,
            scaler_min=app.state.scaler_min,
            scaler_max=app.state.scaler_max,
        )

        if result.get("status") == "error":
            logger.error("Prediction pipeline error: %s", result.get("message"))
            raise HTTPException(status_code=500, detail=result.get("message", "Prediction failed"))

        return result

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Unhandled exception in /predict: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")