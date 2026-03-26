import joblib
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from collections import deque, defaultdict

model = joblib.load("model.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# store last 1000 scans
scan_log = deque(maxlen=1000)

class EmailRequest(BaseModel):
    text: str
    attachment: str = "No"
    sender: str = ""
    subject: str = ""
    reported: bool = False

@app.post("/classify")
async def classify_email(request: EmailRequest):
    input_df = pd.DataFrame([{
        "Text": request.text,
        "Attachment": request.attachment
    }])

    label = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    score = round(float(max(proba)), 2)

    scan_log.appendleft({
        "id": len(scan_log) + 1,
        "timestamp": datetime.utcnow().isoformat(),
        "sender": request.sender,
        "subject": request.subject,
        "label": label,
        "score": score,
        "reported": request.reported,
        "attachment": request.attachment,
        "body_preview": request.text[:500],   # FIXED
    })

    return {
        "score": score,
        "label": label,
        "attachment": request.attachment,
    }

@app.get("/dashboard/stats")
async def dashboard_stats():
    logs = list(scan_log)
    by_label = {}
    reported_count = 0
    for entry in logs:
        lbl = entry.get("label", "unknown")
        by_label[lbl] = by_label.get(lbl, 0) + 1
        if entry.get("reported"):
            reported_count += 1

    return {
        "total": len(logs),
        "by_label": by_label,
        "reported": reported_count,
        "recent": logs[:20],
    }

@app.get("/dashboard/heatmap")
async def dashboard_heatmap():
    heat = [[0 for _ in range(24)] for _ in range(7)]
    for entry in scan_log:
        if entry["label"] not in ("phishing", "spam"):
            continue
        dt = datetime.fromisoformat(entry["timestamp"])
        heat[dt.weekday()][dt.hour] += 1
    return {"matrix": heat}

@app.get("/dashboard/trends")
async def dashboard_trends(days: int = 14):
    cutoff = datetime.utcnow() - timedelta(days=days)
    per_day = defaultdict(lambda: {"ham": 0, "spam": 0, "phishing": 0, "support": 0, "reported": 0})

    for entry in scan_log:
        dt = datetime.fromisoformat(entry["timestamp"])
        if dt < cutoff:
            continue
        key = dt.date().isoformat()
        lbl = entry["label"]
        per_day[key][lbl] += 1
        if entry["reported"]:
            per_day[key]["reported"] += 1

    days_sorted = sorted(per_day.keys())
    return {
        "days": days_sorted,
        "data": [per_day[d] for d in days_sorted],
    }

@app.get("/dashboard/leaderboard")
async def dashboard_leaderboard():
    stats = {}
    for entry in scan_log:
        sender = entry.get("sender") or "Unknown"
        if sender not in stats:
            stats[sender] = {"scans": 0, "reported": 0}
        stats[sender]["scans"] += 1
        if entry["reported"]:
            stats[sender]["reported"] += 1

    rows = sorted(
        [{"sender": s, "scans": v["scans"], "reported": v["reported"]} for s, v in stats.items()],
        key=lambda x: x["reported"],
        reverse=True
    )
    return {"rows": rows[:20]}

@app.get("/dashboard/alerts")
async def dashboard_alerts():
    now = datetime.utcnow()
    window = now - timedelta(minutes=10)
    phishing_count = sum(
        1 for e in scan_log
        if e["label"] == "phishing" and datetime.fromisoformat(e["timestamp"]) >= window
    )
    if phishing_count >= 5:
        return {
            "active": True,
            "message": f"Spike detected: {phishing_count} phishing emails in last 10 minutes."
        }
    return {"active": False}

@app.get("/dashboard/export")
async def dashboard_export(label: str | None = Query(default=None)):
    rows = [e for e in scan_log if not label or e["label"] == label]

    lines = ["timestamp,sender,subject,label,score,reported,attachment"]
    for e in rows:
        lines.append(",".join([
            e["timestamp"],
            (e["sender"] or "").replace(",", " "),
            (e["subject"] or "").replace(",", " "),
            e["label"],
            str(e["score"]),
            "1" if e["reported"] else "0",
            e["attachment"],
        ]))

    return PlainTextResponse("\n".join(lines), media_type="text/csv")

@app.get("/")
async def root():
    return FileResponse("index.html")
