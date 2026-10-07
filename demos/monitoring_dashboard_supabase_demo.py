from __future__ import annotations

# region book:monitoring-dashboard-supabase-demo
import os

from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.get("/metrics/agents")
def get_agent_metrics():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise HTTPException(status_code=503, detail="Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
    try:
        from supabase import create_client
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail="supabase is not installed") from exc
    try:
        supabase = create_client(url, key)
        res = supabase.table("agent_metrics").select("*").execute()
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Agent metrics query failed") from exc
    return {"agents": res.data}


# endregion book:monitoring-dashboard-supabase-demo
