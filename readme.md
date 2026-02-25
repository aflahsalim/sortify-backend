# Sortify Backend

Sortify is an email‑safety classification system designed for Outlook (Web & Desktop).  
It helps users quickly understand whether an email is Safe, Suspicious, Spam, or Phishing — directly inside the Outlook task pane.

The system is composed of two parts:
1. Sortify Outlook Add‑in (client)
2. Sortify Backend (optional ML inference service)

This repository contains the backend component.

============================================================
1. WHAT IS SORTIFY?
============================================================

Sortify is a privacy‑respecting email‑risk detection tool built for Outlook.  
It analyzes incoming emails using:
- Local heuristics (built into the add‑in)
- Optional machine‑learning inference (this backend)

Sortify displays:
- A risk gauge (0–100)
- A color‑coded category (Safe, Support, Spam, Phishing)
- A short analysis panel (links, attachments, urgency, sender reputation)
- A confirmation popup for forwarding suspicious emails to support

Sortify does NOT:
- Store email content
- Log messages
- Send full emails to external servers

============================================================
2. ROLE OF THIS REPOSITORY (BACKEND)
============================================================

This backend provides:
- A lightweight ML model
- A simple REST API endpoint
- A fast scoring pipeline (<500 ms)
- A privacy‑respecting inference process

The backend is **optional**.  
If disabled, the add‑in falls back to local heuristics.

============================================================
3. FEATURES
============================================================

- Computes a numeric risk score (0–100)
- Maps results to four categories:
  Safe, Support, Spam, Phishing
- Accepts minimal JSON payload:
  sender domain, link count, attachment count, urgency score, text summary
- Returns optional:
  confidence, explanation flags
- Lightweight, fast, and easy to run locally

============================================================
4. INSTALLATION & SETUP
============================================================

1. Clone the repository:
git clone https://github.com/yourusername/sortify-backend
cd sortify-backend

2. Install dependencies:
pip install -r requirements.txt

3. Run the backend server (choose one):
uvicorn main:app --reload
OR
python app.py

4. Connect the add‑in:
Enter this URL inside the Sortify add‑in settings:
http://localhost:8000/api/infer

If unreachable → fallback to local heuristics.

============================================================
5. API CONTRACT
============================================================

POST /api/infer

Request JSON:
- sender_domain
- body_text or summary
- has_links
- link_count
- has_attachments
- attachment_count
- urgency_score

Response JSON:
- risk_score (0–100)
- category (Safe | Support | Spam | Phishing)
- confidence (optional)
- explanation (optional)

============================================================
6. MODEL DETAILS
============================================================

- Trained offline using labeled email samples
- Lightweight scikit‑learn model
- Loaded into memory at startup
- No continuous training pipeline
- No email content stored or logged

============================================================
7. PRIVACY & SECURITY
============================================================

- No email content stored
- No logs
- Only minimal metadata processed
- HTTPS recommended for production
- Input validation included

============================================================
8. PURPOSE
============================================================

To provide a simple, optional ML inference service that enhances Sortify’s classification accuracy while respecting privacy and operating within Outlook’s constraints.

============================================================
END OF README
============================================================
