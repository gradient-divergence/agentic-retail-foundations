"""
Data models specific to API interactions (e.g., Gateway, Services).
"""

from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

# Authentication and request models (from notebook)
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    agent_id: Optional[str] = None
    roles: List[str] = []

class Agent(BaseModel):
    """Represents an Agent in the context of the API Gateway/Auth."""
    agent_id: str
    agent_name: str
    roles: List[str]
    is_active: bool = True

class RequestLogEntry(BaseModel):
    """Model for logging API Gateway requests."""
    request_id: str
    timestamp: datetime
    method: str
    path: str
    agent_id: Optional[str]
    service: str
    status_code: int
    response_time_ms: float
    error: Optional[str] = None 