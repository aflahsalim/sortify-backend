from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your Outlook add-in frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to your Render domain later
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailRequest(BaseModel):
    text: str

@app.post("/classify")
async def classify_email(request: EmailRequest):
    email_text = request.text

    # 🔍 Replace this with your real ML classifier logic
    # For now, simple mock rules:
    score = 0.91 if "verify" in email_text.lower() else 0.05
    label = "phishing" if score > 0.8 else "safe"

    # 🧠 Risk factor analysis (mocked for now)
    sender_reputation = "Low / Unverified" if "outlook.com" in email_text else "Trusted"
    link_analysis = "Suspicious Redirects" if "http" in email_text else "No Links Found"
    content_check = "Urgency Patterns" if "verify" in email_text.lower() else "Normal Language"

    return {
        "score": score,          # float between 0 and 1
        "label": label,          # "phishing" or "safe"
        "sender": sender_reputation,
        "links": link_analysis,
        "content": content_check
    }
