"""FastAPI application -- language feedback endpoint."""

import logging
import time

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.feedback import get_feedback
from app.models import FeedbackRequest, FeedbackResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Language Feedback API",
    description=(
        "Analyzes learner-written sentences and provides structured language feedback "
        "powered by LLMs. Supports any language and writing system."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Middleware: log response times
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_response_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    logger.info("%s %s completed in %.2fs", request.method, request.url.path, elapsed)
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Analyze a learner sentence and return structured correction feedback."""
    try:
        return await get_feedback(request)
    except Exception as exc:
        logger.error("Feedback failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to generate feedback: {str(exc)}"},
        )
