import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("health_data.csv")

# Risk classification
def classify_risk(cases):
    if cases < 100:
        return "Low"
    elif cases <= 200:
        return "Medium"
    else:
        return "High"

# Create Risk_Level column
data["Risk_Level"] = data["Cases"].apply(classify_risk)

print("\nDataset Preview:")
print(data.head())

print("\nTotal Rows:", len(data))

# Label Encoding
encoder = LabelEncoder()

data["State_Encoded"] = encoder.fit_transform(data["State"])

print("\nEncoded States:")
print(data[["State", "State_Encoded"]].drop_duplicates())

from sklearn.model_selection import train_test_split

# ------------------------
# Features and Target
# ------------------------

X = data[["State_Encoded", "Month", "Cases"]]

y = data["Risk_Level"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# ------------------------
# Train Test Split
# ------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Rows:", len(X_train))
print("Testing Rows:", len(X_test))

# ------------------------
# Train Model
# ------------------------

model = RandomForestClassifier()

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ------------------------
# Model Accuracy
# ------------------------

accuracy = model.score(X_test, y_test)

print("\nModel Accuracy:", accuracy)

# ------------------------
# Prediction Function
# ------------------------

def predict_risk(state, month, cases):

    state_encoded = encoder.transform([state])[0]

    prediction = model.predict([[state_encoded, month, cases]])

    return prediction[0]

# Example Prediction

result = predict_risk("Kerala", 7, 320)

print("\nPrediction Example:")
print("Kerala | Month 7 | Cases 320")
print("Predicted Risk:", result)