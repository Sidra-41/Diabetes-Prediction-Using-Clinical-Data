
# PROJECT NAME: DISEASE PREDICTION


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)

# 1. LOAD DATA


columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv("pima-indians-diabetes.csv", names=columns)

print("\nDataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


# 2. BASIC DATA CLEANING


zero_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[zero_missing] = df[zero_missing].replace(0, np.nan)

print("\nMissing values after replacing invalid zeros:")
print(df.isnull().sum())

# Fill missing values with median
for column in zero_missing:
    df[column] = df[column].fillna(df[column].median())


# 3. EXPLORATORY DATA ANALYSIS

print("\nDescriptive statistics:")
print(df.describe())

print("\nOutcome distribution:")
print(df["Outcome"].value_counts())

# Outcome plot
sns.countplot(x="Outcome", data=df)
plt.title("Diabetes Outcome Distribution")
plt.xlabel("Diabetes (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Clinical Variables")
plt.show()

# 4. SEPARATE FEATURES AND TARGET

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 5. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 6. LOGISTIC REGRESSION

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]

print("\n========== LOGISTIC REGRESSION ==========")

print("Accuracy:",
      accuracy_score(y_test, logistic_predictions))

print("Precision:",
      precision_score(y_test, logistic_predictions))

print("Recall:",
      recall_score(y_test, logistic_predictions))

print("F1 Score:",
      f1_score(y_test, logistic_predictions))

print("ROC-AUC:",
      roc_auc_score(y_test, logistic_probabilities))

print("\nClassification Report:")
print(classification_report(y_test, logistic_predictions))


# 7. RANDOM FOREST

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)
rf_probabilities = random_forest.predict_proba(X_test)[:, 1]

print("\n========== RANDOM FOREST ==========")

print("Accuracy:",
      accuracy_score(y_test, rf_predictions))

print("Precision:",
      precision_score(y_test, rf_predictions))

print("Recall:",
      recall_score(y_test, rf_predictions))

print("F1 Score:",
      f1_score(y_test, rf_predictions))

print("ROC-AUC:",
      roc_auc_score(y_test, rf_probabilities))

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))


# 8. CONFUSION MATRIX

cm = confusion_matrix(y_test, rf_predictions)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 9. ROC CURVE

RocCurveDisplay.from_predictions(
    y_test,
    logistic_probabilities,
    name="Logistic Regression"
)

RocCurveDisplay.from_predictions(
    y_test,
    rf_probabilities,
    name="Random Forest"
)

plt.title("ROC Curves")
plt.show()


# 10. FEATURE IMPORTANCE

importance = pd.Series(
    random_forest.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature importance:")
print(importance)

importance.plot(kind="bar")
plt.title("Clinical Feature Importance")
plt.xlabel("Clinical Feature")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 11. SAVE THE MODEL

joblib.dump(
    random_forest,
    "diabetes_prediction_model.pkl"
)

print("\nModel saved as diabetes_prediction_model.pkl")


# 12. TEST WITH A NEW PATIENT

new_patient = pd.DataFrame({
    "Pregnancies": [2],
    "Glucose": [140],
    "BloodPressure": [80],
    "SkinThickness": [25],
    "Insulin": [100],
    "BMI": [30],
    "DiabetesPedigreeFunction": [0.5],
    "Age": [35]
})

prediction = random_forest.predict(new_patient)[0]
probability = random_forest.predict_proba(new_patient)[0][1]

print("\n========== NEW PATIENT ==========")

if prediction == 1:
    print("Predicted outcome: Higher predicted risk of diabetes")
else:
    print("Predicted outcome: Lower predicted risk of diabetes")

print("Predicted probability:",
      round(probability, 3))

print("\nNote: This model is for educational/research purposes")
print("and is NOT a medical diagnostic tool.")