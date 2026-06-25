import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

df = pd.read_csv("placement.csv")

gender_encoder = LabelEncoder()
stream_encoder = LabelEncoder()

df["Gender"] = gender_encoder.fit_transform(df["Gender"])
df["Stream"] = stream_encoder.fit_transform(df["Stream"])

X = df.drop("PlacedOrNot", axis=1)
y = df["PlacedOrNot"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Best model for smooth probabilities
model = LogisticRegression(max_iter=1000, random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_[0]
})

print("\nFeature Importance:")
print(
    feature_importance.sort_values(
        by="Importance",
        ascending=False
    )
)

joblib.dump(model, "placement_model.pkl")
joblib.dump(gender_encoder, "gender_encoder.pkl")
joblib.dump(stream_encoder, "stream_encoder.pkl")

print("\nModel saved successfully")