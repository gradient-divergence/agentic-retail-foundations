from __future__ import annotations

# region book:api-gateway-imports
import logging
import time
import uuid
from datetime import datetime

import httpx

try:
    import redis
except ModuleNotFoundError:
    raise SystemExit("Install the streaming extra: uv sync --extra streaming") from None
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# endregion book:api-gateway-imports


# region book:api-gateway-config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retail-api-gateway")

app = FastAPI(
    title="Retail Agent API Gateway",
    description="Centralized gateway for retail agent communication",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# endregion book:api-gateway-config


# region book:api-gateway-redis-config
redis_client = redis.Redis(host="redis", port=6379, db=0)

SERVICE_REGISTRY = {
    "product-service": "http://product-service:8000",
    "inventory-service": "http://inventory-service:8001",
    "order-service": "http://order-service:8002",
    "customer-service": "http://customer-service:8003",
    "pricing-service": "http://pricing-service:8004",
}
# endregion book:api-gateway-redis-config


# region book:api-gateway-token
class Token(BaseModel):
    access_token: str
    token_type: str


# endregion book:api-gateway-token


# region book:api-gateway-token-data
class TokenData(BaseModel):
    agent_id: str | None = None
    roles: list[str] = []


# endregion book:api-gateway-token-data


# region book:api-gateway-agent-model
class Agent(BaseModel):
    agent_id: str
    agent_name: str
    roles: list[str]
    is_active: bool = True


# endregion book:api-gateway-agent-model


# region book:api-gateway-request-log
class RequestLogEntry(BaseModel):
    request_id: str
    timestamp: datetime
    method: str
    path: str
    agent_id: str | None
    service: str
    status_code: int
    response_time_ms: float
    error: str | None = None


# endregion book:api-gateway-request-log


# region book:api-gateway-rate-limits
RATE_LIMITS = {
    "default": 100,
    "inventory-agent": {"default": 200, "/api/inventory": 500},
    "pricing-agent": {"default": 300, "/api/pricing/batch-update": 50},
}
# endregion book:api-gateway-rate-limits


async def log_request(entry: RequestLogEntry) -> None:
    logger.info("Gateway request %s %s %s", entry.request_id, entry.method, entry.path)


# region book:api-gateway-proxy-signature
async def proxy_request(
    request: Request,
    service: str,
    path: str,
    agent: Agent,
    background_tasks: BackgroundTasks,
):
    if service not in SERVICE_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Service {service} not found")

    service_url = SERVICE_REGISTRY[service]
    target_url = f"{service_url}{path}"
    # endregion book:api-gateway-proxy-signature

    # region book:api-gateway-proxy-timing
    start_time = time.time()
    request_id = str(uuid.uuid4())
    # endregion book:api-gateway-proxy-timing

    # region book:api-gateway-proxy-headers
    method = request.method
    headers = dict(request.headers)
    headers["X-Retail-Gateway-RequestId"] = request_id
    headers["X-Retail-Agent-Id"] = agent.agent_id
    headers["X-Retail-Agent-Roles"] = ",".join(agent.roles)
    # endregion book:api-gateway-proxy-headers

    # region book:api-gateway-proxy-strip-headers
    for header in ["host", "content-length"]:
        if header in headers:
            del headers[header]
    # endregion book:api-gateway-proxy-strip-headers

    # region book:api-gateway-proxy-body
    body = await request.body()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(method, target_url, headers=headers, content=body, timeout=30.0)

        request_time_ms = (time.time() - start_time) * 1000
        log_entry = RequestLogEntry(
            request_id=request_id,
            timestamp=datetime.now(),
            method=method,
            path=path,
            agent_id=agent.agent_id,
            service=service,
            status_code=response.status_code,
            response_time_ms=request_time_ms,
        )
        background_tasks.add_task(log_request, log_entry)

        return JSONResponse(
            content=response.json() if response.content else None,
            status_code=response.status_code,
            headers=dict(response.headers),
        )
    except Exception as exc:
        request_time_ms = (time.time() - start_time) * 1000
        log_entry = RequestLogEntry(
            request_id=request_id,
            timestamp=datetime.now(),
            method=method,
            path=path,
            agent_id=agent.agent_id,
            service=service,
            status_code=500,
            response_time_ms=request_time_ms,
            error=str(exc),
        )
        background_tasks.add_task(log_request, log_entry)
        raise HTTPException(status_code=500, detail=f"Service error: {str(exc)}") from exc


# endregion book:api-gateway-proxy-body
