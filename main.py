import joblib
import sqlite3
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

# ── App setup ─────────────────────────────────────────────────────
app = FastAPI()

# Allow requests from Outlook add-in (any origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained ML model once at startup
model = joblib.load("model.pkl")

# ── Database setup ────────────────────────────────────────────────
DB = "sortify.db"

def get_db():
    """Open a SQLite connection with row-as-dict support."""
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create the scans table if it doesn't exist yet."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT,
            sender           TEXT,
            subject          TEXT,
            label            TEXT,
            sender_risk      TEXT,
            auth_result      TEXT,
            files_result     TEXT,
            urgency_result   TEXT,
            attachment_count INTEGER DEFAULT 0,
            reported         INTEGER DEFAULT 0,
            body_preview     TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()   # runs every startup, safe to call multiple times

# ── Request models ────────────────────────────────────────────────

class EmailRequest(BaseModel):
    """Payload for the ML classify endpoint."""
    text: str
    attachment: str = "No"

class ScanLog(BaseModel):
    """Payload sent by the add-in for every scan and every report."""
    sender:           Optional[str] = ""
    subject:          Optional[str] = ""
    label:            Optional[str] = "unknown"
    sender_risk:      Optional[str] = ""
    auth_result:      Optional[str] = ""
    files_result:     Optional[str] = ""
    urgency_result:   Optional[str] = ""
    attachment_count: Optional[int] = 0
    reported:         Optional[bool] = False
    body_preview:     Optional[str] = ""

# ── Helper ────────────────────────────────────────────────────────

def insert_scan(data: ScanLog, force_reported: bool = False):
    """Insert one scan row into the database."""
    conn = get_db()
    conn.execute("""
        INSERT INTO scans
        (timestamp, sender, subject, label, sender_risk, auth_result,
         files_result, urgency_result, attachment_count, reported, body_preview)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.utcnow().isoformat(),
        data.sender,   data.subject,  data.label,
        data.sender_risk, data.auth_result,
        data.files_result, data.urgency_result,
        data.attachment_count,
        1 if (data.reported or force_reported) else 0,
        data.body_preview
    ))
    conn.commit()
    conn.close()

# ── Existing endpoint — ML classify ──────────────────────────────
@app.post("/classify")
async def classify_email(request: EmailRequest):
    """Run the ML model and return a risk label + score."""
    input_df = pd.DataFrame([{
        "Text":       request.text,
        "Attachment": request.attachment
    }])
    label = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    score = round(max(proba), 2)

    label_map = {
        "ham":      {"display": "Safe",           "color": "green"},
        "spam":     {"display": "Spam",            "color": "orange"},
        "phishing": {"display": "Phishing Risk",   "color": "red"},
        "support":  {"display": "Support Ticket",  "color": "blue"},
    }
    mapped = label_map.get(label, {"display": label, "color": "gray"})
    return {
        "score":   score,
        "label":   label,
        "display": mapped["display"],
        "color":   mapped["color"]
    }

# ── New endpoint — log every scan silently ────────────────────────
@app.post("/log-scan")
async def log_scan(data: ScanLog):
    """Called automatically by the add-in after every classification."""
    insert_scan(data)
    return {"status": "logged"}

# ── New endpoint — user-triggered report ──────────────────────────
@app.post("/report")
async def report_email(data: ScanLog):
    """Called when user clicks 'Send to Sortify team'. Flags as reported."""
    insert_scan(data, force_reported=True)
    return {"status": "reported"}

# ── New endpoint — stats for the dashboard ───────────────────────
@app.get("/dashboard/stats")
async def dashboard_stats():
    """Returns scan counts and recent rows for the admin dashboard."""
    conn = get_db()

    total    = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    reported = conn.execute("SELECT COUNT(*) FROM scans WHERE reported=1").fetchone()[0]

    # Count by label
    by_label = conn.execute(
        "SELECT label, COUNT(*) as count FROM scans GROUP BY label"
    ).fetchall()

    # Last 20 scans for the table
    recent = conn.execute(
        "SELECT timestamp, sender, subject, label, reported FROM scans ORDER BY id DESC LIMIT 20"
    ).fetchall()

    conn.close()
    return {
        "total":    total,
        "reported": reported,
        "by_label": {r["label"]: r["count"] for r in by_label},
        "recent":   [dict(r) for r in recent]
    }

# ── New endpoint — serve the admin dashboard page ─────────────────
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serves the admin dashboard HTML page."""
    with open("index.html", "r") as f:
        return f.read()
