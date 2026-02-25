Sortify Backend

Sortify Backend provides the optional machine‑learning inference service used by the Sortify Outlook add‑in.  
It receives minimal email‑derived features, runs a lightweight ML model, and returns a risk score and category.  
The backend is not required for the add‑in to function; local heuristics can be used when the backend is disabled.

---

## Features
- Computes a numeric risk score (0–100)
- Maps results to four categories: **Safe**, **Support**, **Spam**, **Phishing**
- Accepts minimal JSON payload (links, attachments, urgency, sender domain, text summary)
- Returns optional confidence and explanation flags
- Lightweight, fast, and easy to run locally

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/sortify-backend
cd sortify-backend
2. Install dependencies
bash
pip install -r requirements.txt
3. Run the backend server
Depending on your implementation:

bash
uvicorn main:app --reload
or

bash
python app.py
4. Connect the add‑in
Inside the Sortify add‑in settings, enter your backend URL:

Code
http://localhost:8000/api/infer
If the backend is unreachable, the add‑in automatically falls back to local heuristics.

API Contract
POST /api/infer
Request JSON fields:

sender_domain

body_text or summary

has_links

link_count

has_attachments

attachment_count

urgency_score

Response JSON fields:

risk_score (0–100)

category (Safe | Support | Spam | Phishing)

confidence (optional)

explanation (optional)

Tech Stack
Python 3.x

FastAPI or Flask

Scikit‑learn (or similar)

Purpose
To provide a simple, optional ML inference service that enhances Sortify’s classification accuracy while respecting privacy and operating within the constraints of the Outlook add‑in environment.

============================================================
README 2 — sortify-addin
============================================================

Sortify Outlook Add‑in
Sortify is an Outlook add‑in that evaluates the safety of incoming emails and presents the result directly inside the Outlook task pane.
It uses local heuristics and, optionally, a Python backend to classify emails into four categories: Safe, Support, Spam, and Phishing.

Installation & Setup
1. Download the Manifest File
Download the file:

Code
manifest.xml
2. Install the Add‑in in Outlook
Outlook Web
Open Outlook in your browser

Go to Settings → View all Outlook settings

Navigate to Mail → Customize actions → Add‑ins

Select Add a custom add‑in

Choose Add from file

Upload manifest.xml

Outlook Desktop
Open Outlook

Go to Home → Get Add‑ins

Open My Add‑ins

Scroll to Custom Add‑ins

Select Add from file

Upload manifest.xml

Once installed, Sortify will appear in the Outlook ribbon and task pane.

How to Use
Open any email in Outlook

Open the Sortify task pane

The add‑in automatically:

Reads available email metadata

Computes a local heuristic score

Or calls the backend (if enabled)

The gauge and analysis panel appear instantly

To escalate an email, click Forward to Support and confirm in the popup

Sortify does not store or log email content.

Optional: Enable Backend Mode
If you want to use the machine‑learning backend:

Run the Sortify backend locally

Open the add‑in settings

Enter your backend URL (e.g., http://localhost:8000/api/infer)

Save settings

If the backend is unavailable, Sortify automatically falls back to local heuristics.

Features
Semicircular gauge showing risk percentage and color‑coded category

Analysis panel with:

Sender reputation (Trusted / Unknown)

Link presence

Attachment presence

Urgency level

Optional backend inference for ML‑based scoring

Confirmation popup for forwarding emails to support

Graceful fallback values when data is missing

Tech Stack
HTML, CSS, JavaScript

Office.js (Microsoft Office JavaScript APIs)

Optional backend integration via REST API

Purpose
To help users quickly understand the safety of an email by providing a clear, compact, and privacy‑respecting risk assessment directly within Outlook.
