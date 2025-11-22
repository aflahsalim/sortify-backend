from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")

class Email(BaseModel):
    text: str

@app.post("/classify")
def classify(email: Email):
    prediction = model.predict([email.text])[0]
    try:
        score = model.predict_proba([email.text])[0].max()
    except AttributeError:
        score = 1.0
    return {"score": round(float(score), 2), "label": prediction}
