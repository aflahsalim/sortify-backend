import joblib
import pandas as pd
import sqlite3
import os
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta

# ============================
#  DATABASE SETUP
# ============================

DB_PATH = "/home/data.db"  # Azure App Service persistent directory


def init_db():
  conn = sqlite3.connect(DB_PATH)
  cur = conn.cursor()
  cur.execute(
    """
    CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        sender TEXT,
        subject TEXT,
        label TEXT,
        score REAL,
        reported INTEGER,
        attachment TEXT,
        body_preview TEXT
    )
    """
  )
  conn.commit()
  conn.close()


init_db()


def db_execute(query, params=()):
  conn = sqlite3.connect(DB_PATH)
  cur = conn.cursor()
  cur.execute(query, params)
  conn.commit()
  conn.close()


def db_query(query, params=()):
  conn = sqlite3.connect(DB_PATH)
  cur = conn.cursor()
  cur.execute(query, params)
  rows = cur.fetchall()
  conn.close()
  return rows


# ============================
#  FASTAPI SETUP
# ============================

model = joblib.load("model.pkl")

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
)


class EmailRequest(BaseModel):
  text: str
  attachment: str = "No"
  sender: str = ""
  subject: str = ""
  reported: bool = False


# ============================
#  CLASSIFY ENDPOINT
# ============================

@app.post("/classify")
async def classify_email(request: EmailRequest):
  input_df = pd.DataFrame(
    [
      {
        "Text": request.text,
        "Attachment": request.attachment,
      }
    ]
  )

  label = model.predict(input_df)[0]
  proba = model.predict_proba(input_df)[0]
  score = round(float(max(proba)), 2)

  db_execute(
    """
    INSERT INTO scans (timestamp, sender, subject, label, score, reported, attachment, body_preview)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
      datetime.utcnow().isoformat(),
      request.sender,
      request.subject,
      label,
      score,
      1 if request.reported else 0,
      request.attachment,
      request.text[:500],
    ),
  )

  return {
    "score": score,
    "label": label,
    "attachment": request.attachment,
  }


# ============================
#  DASHBOARD ENDPOINTS
# ============================

@app.get("/dashboard/stats")
async def dashboard_stats():
  all_rows = db_query("SELECT label, reported FROM scans")

  by_label = {}
  reported_count = 0

  for lbl, rep in all_rows:
    by_label[lbl] = by_label.get(lbl, 0) + 1
    if rep == 1:
      reported_count += 1

  recent = db_query(
    """
    SELECT timestamp, sender, subject, label, score, reported, attachment, body_preview
    FROM scans ORDER BY id DESC LIMIT 20
    """
  )

  recent_list = [
    {
      "timestamp": r[0],
      "sender": r[1],
      "subject": r[2],
      "label": r[3],
      "score": r[4],
      "reported": bool(r[5]),
      "attachment": r[6],
      "body_preview": r[7],
    }
    for r in recent
  ]

  return {
    "total": len(all_rows),
    "by_label": by_label,
    "reported": reported_count,
    "recent": recent_list,
  }


@app.get("/dashboard/heatmap")
async def dashboard_heatmap():
  rows = db_query("SELECT timestamp, label FROM scans")

  heat = [[0 for _ in range(24)] for _ in range(7)]

  for ts, lbl in rows:
    if lbl not in ("phishing", "spam"):
      continue
    dt = datetime.fromisoformat(ts)
    heat[dt.weekday()][dt.hour] += 1

  return {"matrix": heat}


@app.get("/dashboard/trends")
async def dashboard_trends(days: int = 14):
  cutoff = datetime.utcnow() - timedelta(days=days)

  rows = db_query(
    """
    SELECT timestamp, label, reported
    FROM scans
    WHERE timestamp >= ?
    """,
    (cutoff.isoformat(),),
  )

  per_day = {}

  for ts, lbl, rep in rows:
    day = ts.split("T")[0]
    if day not in per_day:
      per_day[day] = {
        "ham": 0,
        "spam": 0,
        "phishing": 0,
        "support": 0,
        "reported": 0,
      }
    per_day[day][lbl] += 1
    if rep == 1:
      per_day[day]["reported"] += 1

  days_sorted = sorted(per_day.keys())

  return {
    "days": days_sorted,
    "data": [per_day[d] for d in days_sorted],
  }


@app.get("/dashboard/leaderboard")
async def dashboard_leaderboard():
  rows = db_query(
    """
    SELECT sender, COUNT(*), SUM(reported)
    FROM scans
    GROUP BY sender
    ORDER BY SUM(reported) DESC
    LIMIT 20
    """
  )

  return {
    "rows": [
      {"sender": r[0] or "Unknown", "scans": r[1], "reported": r[2] or 0}
      for r in rows
    ]
  }


@app.get("/dashboard/alerts")
async def dashboard_alerts():
  window = datetime.utcnow() - timedelta(minutes=10)

  rows = db_query(
    """
    SELECT COUNT(*)
    FROM scans
    WHERE label='phishing' AND timestamp >= ?
    """,
    (window.isoformat(),),
  )

  count = rows[0][0]

  if count >= 5:
    return {
      "active": True,
      "message": f"Spike detected: {count} phishing emails in last 10 minutes.",
    }

  return {"active": False}


@app.get("/dashboard/export")
async def dashboard_export(label: str | None = Query(default=None)):
  if label:
    rows = db_query(
      """
      SELECT timestamp, sender, subject, label, score, reported, attachment
      FROM scans WHERE label=?
      """,
      (label,),
    )
  else:
    rows = db_query(
      """
      SELECT timestamp, sender, subject, label, score, reported, attachment
      FROM scans
      """
    )

  lines = ["timestamp,sender,subject,label,score,reported,attachment"]

  for r in rows:
    lines.append(
      ",".join(
        [
          r[0],
          (r[1] or "").replace(",", " "),
          (r[2] or "").replace(",", " "),
          r[3],
          str(r[4]),
          "1" if r[5] else "0",
          r[6],
        ]
      )
    )

  return PlainTextResponse("\n".join(lines), media_type="text/csv")


@app.get("/")
async def root():
  return FileResponse("index.html")
