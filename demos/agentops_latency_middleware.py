from __future__ import annotations

# region book:agentops-latency-middleware
import logging
import time

from fastapi import FastAPI, Request, Response

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest
except ModuleNotFoundError:
    raise SystemExit("Install the monitoring extra: uv sync --extra monitoring") from None

LATENCY = Histogram(
    "agent_api_latency_seconds",
    "Agent API latency",
    ["endpoint"],
)

app = FastAPI()
logger = logging.getLogger(__name__)


@app.middleware("http")
async def monitor(request: Request, call_next):
    start = time.perf_counter()
    try:
        return await call_next(request)
    finally:
        elapsed = time.perf_counter() - start
        LATENCY.labels(request.url.path).observe(elapsed)
        logger.debug("Latency %.4fs for %s", elapsed, request.url.path)


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# endregion book:agentops-latency-middleware
