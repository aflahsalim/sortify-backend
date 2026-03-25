import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from collections import deque

# ── Load model ────────────────────────────────────────────────────────────────
model = joblib.load("model.pkl")

app = FastAPI()

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory scan log (last 100 scans) ───────────────────────────────────────
# Resets on server restart. For persistence, swap with a SQLite/DB write.
scan_log = deque(maxlen=100)

# ── Request / Response models ─────────────────────────────────────────────────
class EmailRequest(BaseModel):
    text: str
    attachment: str = "No"
    sender: str = ""
    subject: str = ""
    reported: bool = False

# ── POST /classify ────────────────────────────────────────────────────────────
@app.post("/classify")
async def classify_email(request: EmailRequest):
    input_df = pd.DataFrame([{
        "Text": request.text,
        "Attachment": request.attachment
    }])

    label = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    score = round(max(proba), 2)

    label_map = {
        "ham":      {"display": "Ham (Safe)",      "color": "green"},
        "spam":     {"display": "Spam",             "color": "orange"},
        "phishing": {"display": "Phishing Risk",    "color": "red"},
        "support":  {"display": "Support Ticket",   "color": "blue"},
    }
    mapped = label_map.get(label, {"display": label, "color": "gray"})

    # Log this scan for the dashboard
    scan_log.appendleft({
        "timestamp": datetime.utcnow().isoformat(),
        "sender":    request.sender,
        "subject":   request.subject,
        "label":     label,
        "score":     score,
        "reported":  request.reported,
    })

    return {
        "score":      score,
        "label":      label,
        "display":    mapped["display"],
        "color":      mapped["color"],
        "attachment": request.attachment,
    }

# ── GET /dashboard/stats ──────────────────────────────────────────────────────
# This is what index.html calls. Returns aggregate counts + recent scans list.
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
        "recent":   logs[:20],   # last 20 for the table
    }

# ── Serve index.html at root (so Azure serves the admin page) ─────────────────
@app.get("/")
async def root():
    return FileResponse("index.html")
