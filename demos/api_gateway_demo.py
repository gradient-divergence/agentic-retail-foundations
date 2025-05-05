"""
API Gateway demonstration using FastAPI.
Includes basic routing, mock authentication, rate limiting placeholders, and logging.

Requires Redis for rate limiting simulation.
Run with: uvicorn demos.api_gateway_demo:app --reload
"""

import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

import httpx  # For making async requests to backend services
from fastapi import FastAPI, Request, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt  # For JWT handling
from passlib.context import CryptContext  # For password hashing (mocked)
import redis.asyncio as redis  # For rate limiting

# Import API models
from models.api import Token, TokenData, Agent, RequestLogEntry

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api-gateway")

# --- Configuration ---

# Create FastAPI application
app = FastAPI(
    title="Retail Agent API Gateway",
    description="Centralized gateway for retail agent communication",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis for rate limiting (requires Redis server running)
redis_client: redis.Redis | None = None

# JWT Configuration
SECRET_KEY = "a_very_secret_key_for_dev_only"  # CHANGE THIS IN PRODUCTION!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password Hashing Context (using bcrypt)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 Scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # Endpoint for token generation

# Service registry - Replace with dynamic discovery (e.g., Consul, Eureka) in production
SERVICE_REGISTRY = {
    "product-service": "http://localhost:8002",  # Example backend service URLs
    "inventory-service": "http://localhost:8001",
    "order-service": "http://localhost:8003",
    # Add other services
}

# Rate limiting configuration (requests per minute)
# More sophisticated implementation needed for actual limiting
RATE_LIMITS = {
    "default": 1000,
    "inventory-agent": {"default": 2000, "/inventory": 5000},
}

# Mock user/agent database (Replace with actual database interaction)
FAKE_AGENTS_DB = {
    "inventory-agent-1": {
        "agent_id": "inventory-agent-1",
        "agent_name": "Inventory Manager Bot",
        "hashed_password": pwd_context.hash("password123"),
        "roles": ["inventory_read", "inventory_write"],
        "is_active": True,
    },
    "pricing-agent-1": {
        "agent_id": "pricing-agent-1",
        "agent_name": "Dynamic Pricer",
        "hashed_password": pwd_context.hash("pass987"),
        "roles": ["pricing_read", "pricing_write"],
        "is_active": True,
    },
}

# --- Startup/Shutdown Events ---


@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection on startup."""
    global redis_client
    try:
        redis_client = redis.Redis(
            host="localhost", port=6379, db=1, decode_responses=True
        )  # Use DB 1
        await redis_client.ping()
        logger.info("Connected to Redis for rate limiting.")
    except Exception as e:
        logger.error(
            f"Failed to connect to Redis for rate limiting: {e}. Rate limiting disabled."
        )
        redis_client = None


@app.on_event("shutdown")
async def shutdown_event():
    """Close Redis connection on shutdown."""
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed.")


# --- Authentication ---


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_agent(agent_id: str) -> Agent | None:
    agent_data = FAKE_AGENTS_DB.get(agent_id)
    if agent_data:
        # Ensure roles is iterable
        roles_data = agent_data.get("roles", [])
        roles_list = (
            [str(r) for r in roles_data] if isinstance(roles_data, list) else []
        )
        return Agent(
            agent_id=str(agent_data["agent_id"]),
            agent_name=str(agent_data["agent_name"]),
            roles=roles_list,  # Use checked list
            is_active=bool(agent_data.get("is_active", True)),
        )
    return None


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)  # Default expiry
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_active_agent(token: str = Depends(oauth2_scheme)) -> Agent:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        agent_id: str | None = payload.get("sub")
        if agent_id is None:
            raise credentials_exception
        token_data = TokenData(agent_id=agent_id, roles=payload.get("roles", []))
    except JWTError:
        raise credentials_exception

    if token_data.agent_id is None:
        raise credentials_exception

    agent = get_agent(agent_id=token_data.agent_id)
    if agent is None:
        raise credentials_exception
    if not agent.is_active:
        raise HTTPException(status_code=400, detail="Inactive agent")
    return agent


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    agent_data = FAKE_AGENTS_DB.get(form_data.username)  # Use .get() for safety
    if not agent_data or not verify_password(
        form_data.password, agent_data["hashed_password"]
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect agent ID or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": agent_data["agent_id"], "roles": agent_data["roles"]},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}


# --- Rate Limiting (Placeholder) ---


async def rate_limiter(
    request: Request, agent: Agent = Depends(get_current_active_agent)
):
    """Placeholder rate limiting dependency."""
    if not redis_client:
        logger.warning("Rate limiting disabled: Redis client not available.")
        return True  # Skip limiting if Redis isn't connected

    # Simple token bucket or fixed window counter needed here
    # Example: Fixed window counter (very basic)
    key = f"rate_limit:{agent.agent_id}:{request.url.path}:{datetime.now().minute}"
    try:
        current_count = await redis_client.incr(key)
        if current_count == 1:
            await redis_client.expire(key, 60)  # Expire key after 60 seconds

        # Get limit for this agent/path safely
        agent_limits = RATE_LIMITS.get(
            agent.agent_id, RATE_LIMITS["default"]
        )  # Fallback to default limit
        if isinstance(agent_limits, dict):
            path_limit = agent_limits.get(
                request.url.path, agent_limits.get("default", RATE_LIMITS["default"])
            )
        else:  # It's the default int limit
            path_limit = agent_limits

        if current_count > path_limit:
            logger.warning(
                f"Rate limit exceeded for agent {agent.agent_id} on {request.url.path}"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
    except Exception as e:
        logger.error(f"Error during rate limiting check: {e}")
        # Fail open (allow request) if rate limiting check fails
        return True
    return True


# --- Request Logging ---


async def log_request(log_entry: RequestLogEntry):
    """Store request logs."""
    # In production: await db.request_logs.insert_one(log_entry.model_dump())
    log_line = f"RID={log_entry.request_id} AID={log_entry.agent_id} {log_entry.method} {log_entry.service}{log_entry.path} -> {log_entry.status_code} ({log_entry.response_time_ms:.2f}ms)"
    if log_entry.error:
        log_line += f" ERR={log_entry.error}"
    logger.info(log_line)


# --- Proxy Logic ---


async def proxy_request(
    request: Request,
    service: str,
    path: str,
    background_tasks: BackgroundTasks,
    agent: Agent = Depends(get_current_active_agent),
    # rate_limit_passed: bool = Depends(rate_limiter) # Apply rate limiter
):
    """Proxy requests to backend services with tracking and logging."""
    # Note: Rate limiter dependency commented out as it needs refinement
    # if not rate_limit_passed: return # Should not happen if Depends raises HTTPException

    if service not in SERVICE_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Service '{service}' not found")

    service_url = SERVICE_REGISTRY[service]
    target_url = f"{service_url}{path}"
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Prepare headers for backend service
    headers = dict(request.headers)
    headers["X-Request-ID"] = request_id  # Pass unique ID
    headers["X-Agent-ID"] = agent.agent_id
    headers["X-Agent-Roles"] = ",".join(agent.roles)
    # Clean up headers not meant for backend
    headers.pop("host", None)
    headers.pop("authorization", None)  # Don't pass gateway token
    headers.pop("content-length", None)

    body = await request.body()
    query_params = request.query_params

    # Add type hints to log_entry_base
    log_entry_base: dict[str, Any] = {
        "request_id": request_id,
        "timestamp": datetime.now(),
        "method": request.method,
        "path": path,
        "agent_id": agent.agent_id,
        "service": service,
        "error": None,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                params=query_params,
                content=body,
                timeout=30.0,
            )
            response.raise_for_status()  # Raise exception for 4xx/5xx responses

            request_time_ms = (time.time() - start_time) * 1000
            # Construct log entry explicitly with type checks/casts
            log_entry = RequestLogEntry(
                request_id=str(log_entry_base["request_id"]),
                timestamp=log_entry_base["timestamp"],
                method=str(log_entry_base["method"]),
                path=str(log_entry_base["path"]),
                agent_id=str(log_entry_base["agent_id"])
                if log_entry_base["agent_id"]
                else None,
                service=str(log_entry_base["service"]),
                status_code=int(response.status_code),
                response_time_ms=float(request_time_ms),
                error=str(log_entry_base["error"]) if log_entry_base["error"] else None,
            )
            background_tasks.add_task(log_request, log_entry)

            # Return raw response to preserve headers, cookies etc.
            return response

    except httpx.HTTPStatusError as exc:
        request_time_ms = (time.time() - start_time) * 1000
        error_detail = f"Backend service error: {exc.response.status_code} {exc.response.text[:100]}"
        # Construct log entry explicitly with type checks/casts
        log_entry = RequestLogEntry(
            request_id=str(log_entry_base["request_id"]),
            timestamp=log_entry_base["timestamp"],
            method=str(log_entry_base["method"]),
            path=str(log_entry_base["path"]),
            agent_id=str(log_entry_base["agent_id"])
            if log_entry_base["agent_id"]
            else None,
            service=str(log_entry_base["service"]),
            status_code=int(exc.response.status_code),
            response_time_ms=float(request_time_ms),
            error=str(error_detail),
        )
        background_tasks.add_task(log_request, log_entry)
        raise HTTPException(status_code=exc.response.status_code, detail=error_detail)

    except Exception as e:
        request_time_ms = (time.time() - start_time) * 1000
        error_detail = f"Gateway or service error: {str(e)}"
        # Construct log entry explicitly
        log_entry = RequestLogEntry(
            request_id=log_entry_base["request_id"],
            timestamp=log_entry_base["timestamp"],
            method=log_entry_base["method"],
            path=log_entry_base["path"],
            agent_id=log_entry_base["agent_id"],
            service=log_entry_base["service"],
            status_code=500,
            response_time_ms=request_time_ms,
            error=error_detail,
        )
        background_tasks.add_task(log_request, log_entry)
        raise HTTPException(status_code=500, detail=error_detail)


# --- API Routes (Example) ---


# Example: Route all /inventory/* requests to inventory-service
@app.api_route("/inventory/{path:path}")
async def inventory_proxy(
    request: Request,
    path: str,
    background_tasks: BackgroundTasks,
    agent: Agent = Depends(get_current_active_agent),
):
    return await proxy_request(
        request, "inventory-service", f"/{path}", background_tasks, agent
    )


# Example: Route all /products/* requests to product-service
@app.api_route("/products/{path:path}")
async def product_proxy(
    request: Request,
    path: str,
    background_tasks: BackgroundTasks,
    agent: Agent = Depends(get_current_active_agent),
):
    return await proxy_request(
        request, "product-service", f"/{path}", background_tasks, agent
    )


# Add routes for other services (orders, pricing, etc.)


# Simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    # Note: Ensure Redis is running for rate limiting simulation
    # Note: Backend services (inventory, product etc.) must be running separately
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Gateway on port 8000
