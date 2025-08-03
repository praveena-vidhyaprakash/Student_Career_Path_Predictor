import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("career_data.csv")

le = LabelEncoder()
df["SubjectInterest"] = le.fit_transform(df["SubjectInterest"])
career_encoder = LabelEncoder()
df["PreferredCareer"] = career_encoder.fit_transform(df["PreferredCareer"])

X = df.drop("PreferredCareer", axis=1)
y = df["PreferredCareer"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(le, "interest_encoder.pkl")
joblib.dump(career_encoder, "career_encoder.pkl")

print("✅ Model and encoders saved successfully.")
