from __future__ import annotations

# region book:agentops-latency-middleware
import logging
import time

from fastapi import FastAPI, Request
from prometheus_client import Histogram, generate_latest

LATENCY = Histogram(
    "agent_api_latency_seconds",
    "Agent API latency",
    ["endpoint"],
)

app = FastAPI()
logger = logging.getLogger(__name__)


@app.middleware("http")
async def monitor(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    LATENCY.labels(request.url.path).observe(elapsed)
    logger.debug("Latency %.4fs for %s", elapsed, request.url.path)
    return response


@app.get("/metrics")
async def metrics():
    return generate_latest().decode()


# endregion book:agentops-latency-middleware
