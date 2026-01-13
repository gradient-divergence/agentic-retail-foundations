from __future__ import annotations

# region book:monitoring-dashboard-supabase-demo
import os

from fastapi import FastAPI, HTTPException
from supabase import create_client

app = FastAPI()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(url, key)


@app.get("/metrics/agents")
def get_agent_metrics():
    res = supabase.table("agent_metrics").select("*").execute()
    if res.error:
        raise HTTPException(status_code=500, detail=res.error.message)
    return {"agents": res.data}


# endregion book:monitoring-dashboard-supabase-demo
