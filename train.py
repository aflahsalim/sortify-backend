import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Load dataset (columns: subject, body, attachment, label)
df = pd.read_csv("email dataset.csv")

print("Columns in dataset:", df.columns)  # Debugging step

# Combine subject + body into one text feature
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

X = df[["text", "attachment"]]  # Features: text + attachment
y = df["label"]                # Target labels: ham, spam, phishing, support

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("text", TfidfVectorizer(), "text"),
    ("attachment", OneHotEncoder(), ["attachment"])
])

# Full model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", MultinomialNB())
])

# Train and save
model.fit(X, y)
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved successfully with 4 labels (ham, spam, phishing, support) and attachment feature.")
