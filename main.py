import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Load trained model
model = joblib.load("model.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from /static
app.mount("/static", StaticFiles(directory="public", html=True), name="static")

class EmailRequest(BaseModel):
    text: str
    attachment: str = "Unknown"

@app.post("/classify")
async def classify_email(request: EmailRequest):
    email_text = request.text
    label = model.predict([email_text])[0]
    proba = model.predict_proba([email_text])[0]
    score = round(max(proba), 2)

    label_map = {
        "ham": {"display": "Ham (Safe)", "color": "green"},
        "spam": {"display": "Spam", "color": "orange"},
        "phishing": {"display": "Phishing Risk", "color": "red"},
        "support": {"display": "Support Ticket", "color": "blue"},
    }
    mapped = label_map.get(label, {"display": label, "color": "gray"})

    return {
        "score": score,
        "label": label,
        "display": mapped["display"],
        "color": mapped["color"],
        "attachment": request.attachment
    }
