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

# last 1000 scans kept in memory
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

    label_map = {
        "ham":      {"display": "Ham (Safe)",      "color": "green"},
        "spam":     {"display": "Spam",            "color": "orange"},
        "phishing": {"display": "Phishing Risk",   "color": "red"},
        "support":  {"display": "Support Ticket",  "color": "blue"},
    }
    mapped = label_map.get(label, {"display": label, "color": "gray"})

    scan_log.appendleft({
        "id":        len(scan_log) + 1,
        "timestamp": datetime.utcnow().isoformat(),
        "sender":    request.sender,
        "subject":   request.subject,
        "label":     label,
        "score":     score,
        "reported":  request.reported,
        "attachment": request.attachment,
    })

    return {
        "score":      score,
        "label":      label,
        "display":    mapped["display"],
        "color":      mapped["color"],
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
        "total":    len(logs),
        "by_label": by_label,
        "reported": reported_count,
        "recent":   logs[:20],
    }

@app.get("/dashboard/heatmap")
async def dashboard_heatmap():
    """
    Returns counts by day_of_week (0=Mon) and hour (0-23) for phishing+spam.
    """
    heat = [[0 for _ in range(24)] for _ in range(7)]
    for entry in scan_log:
        lbl = entry.get("label")
        if lbl not in ("phishing", "spam"):
            continue
        ts = entry.get("timestamp")
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            continue
        dow = dt.weekday()
        hour = dt.hour
        heat[dow][hour] += 1
    return {"matrix": heat}

@app.get("/dashboard/trends")
async def dashboard_trends(days: int = 14):
    """
    Returns per-day counts for labels + reported over last N days.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    per_day = defaultdict(lambda: {"ham": 0, "spam": 0, "phishing": 0, "support": 0, "reported": 0})
    for entry in scan_log:
        ts = entry.get("timestamp")
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            continue
        if dt < cutoff:
            continue
        day_key = dt.date().isoformat()
        lbl = entry.get("label", "ham")
        if lbl in per_day[day_key]:
            per_day[day_key][lbl] += 1
        else:
            per_day[day_key][lbl] = per_day[day_key].get(lbl, 0) + 1
        if entry.get("reported"):
            per_day[day_key]["reported"] += 1

    days_sorted = sorted(per_day.keys())
    return {
        "days": days_sorted,
        "data": [per_day[d] for d in days_sorted],
    }

@app.get("/dashboard/leaderboard")
async def dashboard_leaderboard():
    """
    Simple leaderboard by sender: how many scans + how many reported.
    """
    stats = {}
    for entry in scan_log:
        sender = entry.get("sender") or "Unknown"
        if sender not in stats:
            stats[sender] = {"scans": 0, "reported": 0}
        stats[sender]["scans"] += 1
        if entry.get("reported"):
            stats[sender]["reported"] += 1

    rows = [
        {"sender": s, "scans": v["scans"], "reported": v["reported"]}
        for s, v in stats.items()
    ]
    rows.sort(key=lambda r: r["reported"], reverse=True)
    return {"rows": rows[:20]}

@app.get("/dashboard/alerts")
async def dashboard_alerts():
    """
    Simple spike detection: phishing in last 10 minutes.
    """
    now = datetime.utcnow()
    window = now - timedelta(minutes=10)
    phishing_count = 0
    for entry in scan_log:
        if entry.get("label") != "phishing":
            continue
        ts = entry.get("timestamp")
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            continue
        if dt >= window:
            phishing_count += 1

    threshold = 5
    if phishing_count >= threshold:
        return {
            "active": True,
            "type": "phishing_spike",
            "count": phishing_count,
            "message": f"Spike detected: {phishing_count} phishing emails in last 10 minutes.",
        }
    return {"active": False}

@app.get("/dashboard/export")
async def dashboard_export(label: str | None = Query(default=None)):
    """
    Export scans as CSV. Optional ?label=phishing/spam/ham/support.
    """
    rows = []
    for entry in scan_log:
        if label and entry.get("label") != label:
            continue
        rows.append(entry)

    if not rows:
        return PlainTextResponse("timestamp,sender,subject,label,score,reported,attachment\n", media_type="text/csv")

    lines = ["timestamp,sender,subject,label,score,reported,attachment"]
    for e in rows:
        line = [
            e.get("timestamp", ""),
            (e.get("sender") or "").replace(",", " "),
            (e.get("subject") or "").replace(",", " "),
            e.get("label", ""),
            str(e.get("score", "")),
            "1" if e.get("reported") else "0",
            e.get("attachment", ""),
        ]
        lines.append(",".join(line))
    csv_data = "\n".join(lines) + "\n"
    return PlainTextResponse(csv_data, media_type="text/csv")

@app.get("/")
async def root():
    return FileResponse("index.html")
