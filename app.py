import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

df = pd.read_csv("dataset/creditcard.csv")
artifact = joblib.load("model.joblib")
results_df = pd.read_csv("model_results.csv")

model = artifact["model"]
features = artifact["feature_names"]
best_model_name = artifact["model_name"]

total_transactions = len(df)
fraud_transactions = int((df["Class"] == 1).sum())
legitimate_transactions = int((df["Class"] == 0).sum())
fraud_rate = (
    fraud_transactions / total_transactions * 100
    if total_transactions > 0 else 0
)

st.title("💳 Credit Card Fraud Detection")
st.write(
    "Detect potentially fraudulent credit card transactions "
    "using Machine Learning."
)

st.divider()

st.subheader("📊 Transaction Overview")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Legitimate Transactions", f"{legitimate_transactions:,}")
col3.metric("Fraudulent Transactions", f"{fraud_transactions:,}")
col4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

st.divider()

st.subheader("📈 Model Performance")

display_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

st.dataframe(
    results_df[display_columns].style.format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    results_df.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ]
)

st.divider()

st.sidebar.title("Transaction Information")
st.sidebar.write("Enter transaction feature values")

input_values = {}

for feature in features:
    median_value = pd.to_numeric(
        df[feature],
        errors="coerce"
    ).median()

    default_value = (
        float(median_value)
        if pd.notna(median_value)
        else 0.0
    )

    if feature == "Amount":
        input_values[feature] = st.sidebar.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=max(0.0, default_value),
            format="%.6f"
        )
    elif feature == "Time":
        input_values[feature] = st.sidebar.number_input(
            "Transaction Time",
            min_value=0.0,
            value=max(0.0, default_value),
            format="%.6f"
        )
    else:
        input_values[feature] = st.sidebar.number_input(
            feature,
            value=default_value,
            format="%.6f"
        )

input_data = pd.DataFrame(
    [input_values],
    columns=features
)

st.subheader("📝 Transaction Information")
st.dataframe(
    input_data,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.subheader("🔮 Fraud Prediction")

if st.button("Predict Transaction", type="primary"):
    prediction = int(model.predict(input_data)[0])

    st.success("Transaction Prediction Completed!")

    if prediction == 1:
        st.error("⚠️ Potential Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction Classified as Legitimate")

    col1, col2 = st.columns(2)

    col1.metric(
        "Prediction",
        "Fraudulent" if prediction == 1 else "Legitimate"
    )

    fraud_probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        class_labels = list(model.classes_)

        if 1 in class_labels:
            fraud_probability = float(
                probabilities[class_labels.index(1)]
            )
            col2.metric(
                "Fraud Probability",
                f"{fraud_probability * 100:.2f}%"
            )
            st.progress(
                min(max(fraud_probability, 0.0), 1.0)
            )

    result = input_data.copy()
    result["Prediction"] = (
        "Fraudulent" if prediction == 1 else "Legitimate"
    )

    if fraud_probability is not None:
        result["Fraud Probability"] = fraud_probability

    st.subheader("📋 Prediction Details")
    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction",
        data=csv,
        file_name="fraud_prediction.csv",
        mime="text/csv"
    )

st.divider()

st.subheader("📊 Transaction Class Distribution")

class_counts = df["Class"].value_counts().rename(
    index={
        0: "Legitimate",
        1: "Fraudulent"
    }
)

st.bar_chart(class_counts)

st.divider()

st.subheader("📄 Dataset Preview")
st.dataframe(
    df.head(10),
    use_container_width=True,
    hide_index=True
)

st.divider()

st.caption(
    "Credit Card Fraud Detection | Machine Learning | "
    "Logistic Regression, Decision Tree and Random Forest"
)