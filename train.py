import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load dataset
data_path = Path("dataset/creditcard.csv")
df = pd.read_csv(data_path)

print("Dataset Shape:", df.shape)
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nClass Distribution:")
print(df["Class"].value_counts())

# Remove duplicate records
df = df.drop_duplicates()

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"].astype(int)
features = X.columns.tolist()

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create classification models
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=2, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# Train models and evaluate performance
results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# Compare model performance
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by=["F1 Score", "Recall"], ascending=False
).reset_index(drop=True)

print("\nModel Performance:")
print(results_df)

# Select the best model
best_model_name = results_df.loc[0, "Model"]
best_model = models[best_model_name]

# Save model performance results
results_df.to_csv("model_results.csv", index=False)

# Save the best model
model_artifact = {
    "model": best_model,
    "feature_names": features,
    "model_name": best_model_name,
    "target_name": "Class",
    "fraud_label": 1,
    "legitimate_label": 0
}

joblib.dump(model_artifact, "model.joblib")

print("\nBest Model:", best_model_name)
print("Model saved as model.joblib")
print("Results saved as model_results.csv")