import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load your dataset
df = pd.read_csv("email dataset.csv")  # make sure the filename matches exactly

# Create pipeline: vectorizer + classifier
model = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(df['text'], df['label'])

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(model.named_steps['vectorizer'], 'vectorizer.pkl')

print("Model and vectorizer saved successfully.")
