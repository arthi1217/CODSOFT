import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("spam.csv")

# Keep only needed columns 
df = df[["v1", "v2"]]

print(df.head())

# Rename columns
df = df.rename(columns={"v1": "label", "v2": "message"})

# Convert label to numbers
df["label"] = df["label"].map({"spam": 1, "ham": 0})

print(df.head())

# Text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["message"])

y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))