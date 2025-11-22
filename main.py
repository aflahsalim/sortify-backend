from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS so your frontend can call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = joblib.load("model.pkl")

# Health check route
@app.get("/")
def read_root():
    return {"status": "API is running"}

# Define input schema
class Email(BaseModel):
    text: str

# Classification route
@app.post("/classify")
def classify(email: Email):
    try:
        prediction = model.predict([email.text])[0]
        # Optional: include confidence score if model supports predict_proba
        try:
            score = model.predict_proba([email.text])[0].max()
        except AttributeError:
            score = 1.0  # fallback if predict_proba isn't available

        return {
            "score": round(float(score), 2),
            "label": prediction
        }
    except Exception as e:
        return {"error": str(e)}
