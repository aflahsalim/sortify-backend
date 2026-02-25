# Sortify Backend

Sortify Backend provides the optional machine‑learning inference service used by the Sortify Outlook add‑in.  
It receives minimal email‑derived features, runs a lightweight ML model, and returns a risk score and category.  
The backend is not required for the add‑in to function; local heuristics can be used when the backend is disabled.

## Features
- Computes a numeric risk score (0–100)
- Maps results to four categories: Safe, Support, Spam, Phishing
- Accepts minimal JSON payload (links, attachments, urgency, sender domain, text summary)
- Returns optional confidence and explanation flags
- Lightweight, fast, and easy to run locally

## API Contract
### POST `/api/infer`
**Request JSON fields:**
- `sender_domain` (string)
- `body_text` or summary (string)
- `has_links` (bool)
- `link_count` (int)
- `has_attachments` (bool)
- `attachment_count` (int)
- `urgency_score` (int)

**Response JSON fields:**
- `risk_score` (0–100)
- `category` (Safe | Support | Spam | Phishing)
- `confidence` (optional)
- `explanation` (optional short flags)

## Tech Stack
- Python 3.x  
- FastAPI or Flask  
- Scikit‑learn (or similar) for model inference

## Model Notes
- Model is trained offline using labeled email samples  
- No automated retraining pipeline  
- Model artifacts are loaded into memory at startup  
- Backend does **not** store email content or logs

## Purpose
To provide a simple, optional ML inference service that enhances Sortify’s classification accuracy while respecting privacy and operating within the constraints of the Outlook add‑in environment.
