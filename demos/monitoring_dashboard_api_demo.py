from __future__ import annotations

# region book:monitoring-dashboard-api-demo
import os

import psycopg2
from fastapi import FastAPI, HTTPException
from psycopg2.extras import RealDictCursor

app = FastAPI()
DB_URL = os.getenv("SUPABASE_DB_URL")

try:
    conn = psycopg2.connect(DB_URL)
except Exception as exc:
    print("Failed to connect to Supabase DB:", exc)
    conn = None


@app.get("/metrics/agents")
def get_agent_metrics():
    if conn is None:
        raise HTTPException(status_code=500, detail="DB connection not available")
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT agent_id, tasks_completed, avg_response_time, last_updated FROM agent_metrics;")
    rows = cur.fetchall()
    cur.close()
    return {"agents": rows}


# endregion book:monitoring-dashboard-api-demo
