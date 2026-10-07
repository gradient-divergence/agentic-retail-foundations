from __future__ import annotations

# region book:monitoring-dashboard-api-demo
import os

from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.get("/metrics/agents")
def get_agent_metrics():
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise HTTPException(status_code=503, detail="Set SUPABASE_DB_URL to query agent metrics")
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail="psycopg2 is not installed") from exc
    try:
        conn = psycopg2.connect(db_url)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT agent_id, tasks_completed, avg_response_time, last_updated FROM agent_metrics;"
                )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Agent metrics query failed") from exc
    return {"agents": rows}


# endregion book:monitoring-dashboard-api-demo
