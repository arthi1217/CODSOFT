import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("fraudTrain.csv")

# Reduce data size (VERY IMPORTANT for speed)
df = df.sample(n=5000, random_state=42)

print(df.head())

# Remove unnecessary columns
df = df.drop([
    "Unnamed: 0",
    "trans_date_trans_time",
    "cc_num",
    "first",
    "last",
    "street",
    "city",
    "state",
    "zip",
    "job",
    "dob",
    "trans_num"
], axis=1)

print(df.head())

# Convert text to numbers
df = pd.get_dummies(df, drop_first=True)

print(df.head())

# Split input and output
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)

# Train model (FAST)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))