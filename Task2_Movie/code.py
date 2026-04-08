import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("train_data.txt", sep=":::", engine="python")
print(df.head())


df.columns = ["id", "title", "genre", "plot"]
print(df.head())
print(df.columns)

df = df[["plot", "genre"]]
print(df.head())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["plot"])

y = df["genre"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))