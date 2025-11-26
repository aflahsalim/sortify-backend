import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load dataset (must include labels: ham, spam, phishing, support)
df = pd.read_csv("email dataset.csv")

# Create pipeline
model = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train and save
model.fit(df['text'], df['label'])  # labels should be ham, spam, phishing, support
joblib.dump(model, 'model.pkl')

print("Model trained and saved successfully with 4 labels.")
