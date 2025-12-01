import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load trained model
model = joblib.load("model.pkl")

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request schema
class EmailRequest(BaseModel):
    text: str
    attachment: str = "No"  # default to "No" if not provided

@app.post("/classify")
async def classify_email(request: EmailRequest):
    # Build a DataFrame with the same column names used in training
    input_df = pd.DataFrame([{
        "text": request.text,
        "Attachment": request.attachment   # must match training column name exactly
    }])

    # Predict label and confidence
    label = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    score = round(max(proba), 2)

    # Map label to display name and color
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
