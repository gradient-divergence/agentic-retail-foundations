from __future__ import annotations

# region book:api-gateway-imports-joined
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
from fastapi.responses import Response
from pydantic import BaseModel

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


redis_client = redis.Redis(host="redis", port=6379, db=0)

SERVICE_REGISTRY = {
    "product-service": "http://product-service:8000",
    "inventory-service": "http://inventory-service:8001",
    "order-service": "http://order-service:8002",
    "customer-service": "http://customer-service:8003",
    "pricing-service": "http://pricing-service:8004",
}
# endregion book:api-gateway-imports-joined


# region book:api-gateway-token-joined
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    agent_id: str | None = None
    roles: list[str] = []


class Agent(BaseModel):
    agent_id: str
    agent_name: str
    roles: list[str]
    is_active: bool = True


# endregion book:api-gateway-token-joined


# region book:api-gateway-request-log-joined
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


RATE_LIMITS = {
    "default": 100,
    "inventory-agent": {"default": 200, "/api/inventory": 500},
    "pricing-agent": {"default": 300, "/api/pricing/batch-update": 50},
}
# endregion book:api-gateway-request-log-joined


async def log_request(entry: RequestLogEntry) -> None:
    logger.info("gateway.request_completed", extra={"outcome": entry.model_dump(mode="json")})


def _hop_by_hop_headers(headers) -> set[str]:
    blocked = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
    blocked.update(h.strip().lower() for h in headers.get("connection", "").split(","))
    return blocked


def _forward_request_headers(request: Request) -> tuple[set[str], dict[str, str]]:
    blocked = _hop_by_hop_headers(request.headers) | {"host", "content-length"}
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in blocked and not k.lower().startswith("x-retail-")
    }
    return blocked, headers


# region book:api-gateway-proxy-signature-joined
async def proxy_request(
    request: Request,
    service: str,
    path: str,
    agent: Agent,
    background_tasks: BackgroundTasks,
):
    service_url = SERVICE_REGISTRY.get(service)
    target_url = f"{service_url}{path}"
    start_time = time.time()
    request_id = str(uuid.uuid4())
    method = request.method
    _, headers = _forward_request_headers(request)
    headers["X-Retail-Gateway-RequestId"] = request_id
    headers["X-Retail-Agent-Id"] = agent.agent_id
    headers["X-Retail-Agent-Roles"] = ",".join(agent.roles)

    body = await request.body()
    status_code, error = 502, None
    try:
        if service_url is None:
            status_code, error = 404, "unknown_service"
            raise HTTPException(status_code=404, detail="Service not found")
        async with httpx.AsyncClient() as client:
            async with client.stream(
                method,
                httpx.URL(target_url).copy_with(
                    query=request.scope["query_string"],
                ),
                headers=headers,
                content=body,
                timeout=30.0,
            ) as response:
                status_code = response.status_code
                raw_body = b"".join([chunk async for chunk in response.aiter_raw()])
        blocked = _hop_by_hop_headers(response.headers) | {"content-length"}
        forwarded = [(k, v) for k, v in response.headers.multi_items() if k.lower() not in blocked]
        result = Response(content=raw_body, status_code=status_code)
        result.raw_headers.extend((k.encode("latin-1"), v.encode("latin-1")) for k, v in forwarded)
        return result
    except httpx.HTTPError as exc:
        status_code = 502
        error = type(exc).__name__
        raise HTTPException(status_code=502, detail="Upstream service unavailable") from exc
    finally:
        await log_request(
            RequestLogEntry(
                request_id=request_id,
                timestamp=datetime.now(),
                method=method,
                path=path,
                agent_id=agent.agent_id,
                service=service,
                status_code=status_code,
                response_time_ms=(time.time() - start_time) * 1000,
                error=error,
            )
        )


# endregion book:api-gateway-proxy-signature-joined
