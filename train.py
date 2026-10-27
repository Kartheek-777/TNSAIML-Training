from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
RESULTS_PATH = BASE_DIR / "model_results.csv"

RANDOM_STATE = 42


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place creditcard.csv inside the dataset folder."
    )


df = pd.read_csv(DATA_PATH)

if df.empty:
    raise ValueError("The dataset is empty.")

if "Class" not in df.columns:
    raise ValueError("The dataset must contain a 'Class' column.")

print("\n========== DATASET INFORMATION ==========")
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nClass distribution:")
print(df["Class"].value_counts().sort_index())


df = df.drop_duplicates().copy()

if df["Class"].isnull().any():
    raise ValueError("The Class column contains missing values.")

if not set(df["Class"].unique()).issubset({0, 1}):
    raise ValueError("The Class column must contain only 0 and 1.")

df["Class"] = df["Class"].astype(int)

X = df.drop(columns=["Class"])
y = df["Class"]

if X.empty:
    raise ValueError("No input features were found.")

non_numeric = X.select_dtypes(exclude=np.number).columns.tolist()

if non_numeric:
    raise ValueError(f"Non-numeric columns found: {non_numeric}")

X = X.replace([np.inf, -np.inf], np.nan)

if y.value_counts().min() < 2:
    raise ValueError("Each class must have at least two records.")

feature_names = X.columns.tolist()


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n========== DATA SPLITTING ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            feature_names,
        )
    ],
    remainder="drop",
)


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    ),
}


results = []
trained_models = {}


for model_name, classifier in models.items():

    print(f"\n========== TRAINING {model_name} ==========")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "True Negatives": tn,
            "False Positives": fp,
            "False Negatives": fn,
            "True Positives": tp,
        }
    )

    trained_models[model_name] = pipeline

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["F1 Score", "Recall"],
    ascending=False,
).reset_index(drop=True)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(RESULTS_PATH, index=False)


best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]


joblib.dump(
    {
        "model": best_model,
        "feature_names": feature_names,
        "model_name": best_model_name,
        "target_name": "Class",
        "fraud_label": 1,
        "legitimate_label": 0,
    },
    MODEL_PATH,
)


print("\n========== TRAINING COMPLETED ==========")
print("Best model:", best_model_name)
print("Model saved at:", MODEL_PATH)
print("Results saved at:", RESULTS_PATH)