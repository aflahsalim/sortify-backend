import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load trained model
model = joblib.load("model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict later to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request schema
class EmailRequest(BaseModel):
    text: str
    attachment: str = "Unknown"  # Optional field from frontend

# Define classification endpoint
@app.post("/classify")
async def classify_email(request: EmailRequest):
    email_text = request.text

    # Predict label and confidence score
    label = model.predict([email_text])[0]
    proba = model.predict_proba([email_text])[0]
    score = round(max(proba), 2)

    # Map labels to display + color
    label_map = {
        "ham": {"display": "Ham (Safe)", "color": "green"},
        "spam": {"display": "Spam", "color": "orange"},
        "phishing": {"display": "Phishing Risk", "color": "red"},
        "support": {"display": "Support Ticket", "color": "blue"},
    }
    mapped = label_map.get(label, {"display": label, "color": "gray"})

    # Optional analysis details
    sender_reputation = "Low / Unverified" if "outlook.com" in email_text else "Trusted"
    link_analysis = "Suspicious Redirects" if "http" in email_text else "No Links Found"
    content_check = "Urgency Patterns" if any(w in email_text.lower() for w in ["urgent", "verify", "reset", "click"]) else "Normal Language"

    # Return response to frontend
    return {
        "score": score,
        "label": label,
        "display": mapped["display"],
        "color": mapped["color"],
        "sender": sender_reputation,
        "links": link_analysis,
        "content": content_check,
        "attachment": request.attachment
    }
