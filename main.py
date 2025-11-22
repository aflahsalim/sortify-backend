from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your GitHub Pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "API is running"}

# Input schema
class Email(BaseModel):
    text: str

# Classification endpoint
@app.post("/classify")
def classify(email: Email):
    try:
        # Transform the input text using the vectorizer
        X = vectorizer.transform([email.text])
        # Predict using the trained model
        prediction = model.predict(X)[0]
        return {"category": prediction}
    except Exception as e:
        return {"error": str(e)}
