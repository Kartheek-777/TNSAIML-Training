from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.joblib"
DATA_PATH = BASE_DIR / "dataset" / "creditcard.csv"
RESULTS_PATH = BASE_DIR / "model_results.csv"

st.set_page_config(
    page_title="FraudGuard | Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F7FB;
        color: #172B4D;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5EAF2;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #173B73;
    }

    h1, h2, h3 {
        color: #173B73 !important;
        font-weight: 700 !important;
    }

    p, label, .stMarkdown {
        color: #344563;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 3px 12px rgba(23, 59, 115, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #62748E !important;
    }

    [data-testid="stMetricValue"] {
        color: #173B73 !important;
        font-weight: 700;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background-color: #2563EB;
        color: #FFFFFF;
        border: none;
        border-radius: 9px;
        padding: 0.65rem 1.2rem;
        font-weight: 600;
        transition: 0.2s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background-color: #1748B5;
        color: #FFFFFF;
        border: none;
    }

    div[data-baseweb="input"],
    div[data-baseweb="select"] {
        border-radius: 8px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E5EAF2;
        border-radius: 10px;
        overflow: hidden;
    }

    hr {
        border-color: #E1E7F0;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    .page-subtitle {
        color: #62748E;
        font-size: 15px;
        margin-top: -8px;
        margin-bottom: 24px;
    }

    .section-card {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 16px;
    }

    .brand-name {
        color: #2563EB;
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .brand-caption {
        color: #62748E;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)


missing_files = []

if not MODEL_PATH.exists():
    missing_files.append("model.joblib")

if not DATA_PATH.exists():
    missing_files.append("dataset/creditcard.csv")

if not RESULTS_PATH.exists():
    missing_files.append("model_results.csv")

if missing_files:
    st.error(
        "Required project files are missing: "
        + ", ".join(missing_files)
    )
    st.info("Check your project folder and run python train.py if necessary.")
    st.stop()

try:
    artifact = load_model()
    df = load_dataset()
    results_df = load_results()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()

model = artifact["model"]
feature_names = artifact["feature_names"]
best_model_name = artifact["model_name"]

total_transactions = len(df)
fraud_transactions = int((df["Class"] == 1).sum())
legitimate_transactions = int((df["Class"] == 0).sum())

fraud_percentage = (
    fraud_transactions / total_transactions * 100
    if total_transactions > 0
    else 0
)

with st.sidebar:
    st.markdown(
        """
        <div class="brand-name">💳 FraudGuard</div>
        <div class="brand-caption">
            Credit Card Fraud Detection System
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Navigation")

    page_options = [
        "Dashboard",
        "Fraud Detection",
        "Model Performance",
        "Dataset Explorer",
        "About Project",
    ]

    if "navigation_page" in st.session_state:
        default_page = st.session_state.pop("navigation_page")
        default_index = (
            page_options.index(default_page)
            if default_page in page_options
            else 0
        )
    else:
        default_index = 0

    page = st.radio(
        "Choose a page",
        page_options,
        index=default_index,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Model Information")
    st.write("**Selected Model**")
    st.info(best_model_name)

    st.write("**Algorithms Evaluated**")
    st.write("1. Logistic Regression")
    st.write("2. Decision Tree")
    st.write("3. Random Forest")

    st.divider()
    st.caption("Machine Learning Project")
    st.caption("Built with Python and Streamlit")


if page == "Dashboard":
    st.title("Dashboard")

    st.markdown(
        '<div class="page-subtitle">'
        'Monitor transaction data and machine learning performance.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Legitimate Transactions", f"{legitimate_transactions:,}")
    col3.metric("Fraudulent Transactions", f"{fraud_transactions:,}")
    col4.metric("Fraud Rate", f"{fraud_percentage:.3f}%")

    st.write("")
    st.success(
        f"Your selected machine learning model is {best_model_name}."
    )
    st.divider()

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Transaction Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        labels = ["Legitimate", "Fraudulent"]
        values = [legitimate_transactions, fraud_transactions]

        ax.bar(
            labels,
            values,
            color=["#2563EB", "#EF4444"],
            width=0.55,
        )

        ax.set_ylabel("Number of Transactions")
        ax.set_title("Legitimate vs Fraudulent Transactions")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right_col:
        st.subheader("Model F1 Score Comparison")

        if {"Model", "F1 Score"}.issubset(results_df.columns):
            chart_df = results_df.sort_values(
                "F1 Score",
                ascending=True,
            )

            fig, ax = plt.subplots(figsize=(7, 4))

            bar_colors = [
                "#2563EB" if name == best_model_name else "#93C5FD"
                for name in chart_df["Model"]
            ]

            ax.barh(
                chart_df["Model"],
                chart_df["F1 Score"],
                color=bar_colors,
            )

            ax.set_xlabel("F1 Score")
            ax.set_xlim(0, 1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("F1 Score information is unavailable.")

    st.divider()
    st.subheader("Quick Actions")

    action1, action2 = st.columns(2)

    with action1:
        st.markdown("**Check a transaction**")
        st.write(
            "Use the trained machine learning model to predict "
            "whether a transaction may be fraudulent."
        )

        if st.button("Open Fraud Detection", use_container_width=True):
            st.session_state["navigation_page"] = "Fraud Detection"
            st.rerun()

    with action2:
        st.markdown("**Compare algorithms**")
        st.write(
            "Review accuracy, precision, recall, and F1 Score "
            "for all evaluated models."
        )

        if st.button("View Model Performance", use_container_width=True):
            st.session_state["navigation_page"] = "Model Performance"
            st.rerun()


elif page == "Fraud Detection":
    st.title("Fraud Detection")

    st.markdown(
        '<div class="page-subtitle">'
        'Enter transaction feature values to generate a prediction.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "The model requires numerical feature values from the dataset. "
        "Invented values may produce meaningless predictions."
    )

    st.subheader("Transaction Information")

    st.write(
        "Enter the values for the transaction features below. "
        "The form includes all input features expected by the trained model."
    )

    with st.form("transaction_prediction_form"):
        input_values = {}
        input_columns = st.columns(3)

        for index, feature in enumerate(feature_names):
            with input_columns[index % 3]:
                input_values[feature] = st.number_input(
                    label=feature,
                    value=0.0,
                    format="%.6f",
                    key=f"input_{feature}",
                )

        submitted = st.form_submit_button(
            "Predict Transaction",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame(
            [input_values],
            columns=feature_names,
        )

        values_array = input_df.to_numpy(dtype=float)

        if not np.isfinite(values_array).all():
            st.error("Please enter valid numerical values.")
        else:
            try:
                prediction = int(model.predict(input_df)[0])

                st.divider()
                st.subheader("Prediction Result")

                if prediction == 1:
                    st.error("⚠️ Potential Fraud Detected")
                    st.write(
                        "The model classified this transaction "
                        "as potentially fraudulent."
                    )
                else:
                    st.success("✅ Transaction Classified as Legitimate")
                    st.write(
                        "The model classified this transaction "
                        "as legitimate."
                    )

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_df)[0]
                    class_labels = list(model.classes_)

                    if 1 in class_labels:
                        fraud_probability = float(
                            probabilities[class_labels.index(1)]
                        )

                        result_col1, result_col2 = st.columns(2)

                        with result_col1:
                            st.metric(
                                "Estimated Fraud Probability",
                                f"{fraud_probability * 100:.2f}%",
                            )

                        with result_col2:
                            st.metric(
                                "Model Prediction",
                                "Potential Fraud"
                                if prediction == 1
                                else "Legitimate",
                            )

                        st.progress(
                            min(max(fraud_probability, 0.0), 1.0)
                        )

                st.caption(
                    "This is an experimental machine learning prediction. "
                    "It is not a guarantee and should not be used as the "
                    "sole basis for real financial decisions."
                )

            except Exception as error:
                st.error(f"Unable to generate prediction: {error}")


elif page == "Model Performance":
    st.title("Model Performance")

    st.markdown(
        '<div class="page-subtitle">'
        'Evaluate and compare the machine learning algorithms.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Algorithm Comparison")

    display_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in results_df.columns
    ]

    formatted_df = results_df[available_columns].copy()

    numeric_columns = [
        column
        for column in available_columns
        if column != "Model"
    ]

    for column in numeric_columns:
        formatted_df[column] = formatted_df[column].map(
            lambda value: f"{value:.4f}"
        )

    st.dataframe(
        formatted_df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Performance Metrics")

    selected_metric = st.selectbox(
        "Choose a metric",
        ["Accuracy", "Precision", "Recall", "F1 Score"],
    )

    if selected_metric in results_df.columns:
        sorted_results = results_df.sort_values(
            selected_metric,
            ascending=True,
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        bar_colors = [
            "#2563EB" if name == best_model_name else "#93C5FD"
            for name in sorted_results["Model"]
        ]

        ax.barh(
            sorted_results["Model"],
            sorted_results[selected_metric],
            color=bar_colors,
        )

        ax.set_xlabel(selected_metric)
        ax.set_title(f"Model Comparison: {selected_metric}")
        ax.set_xlim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Understanding the Metrics")

    with st.expander("Accuracy"):
        st.write(
            "Accuracy measures the proportion of all predictions "
            "that were correct."
        )

    with st.expander("Precision"):
        st.write(
            "Precision measures how many transactions predicted as "
            "fraudulent were actually fraudulent."
        )

    with st.expander("Recall"):
        st.write(
            "Recall measures how many actual fraudulent transactions "
            "the model successfully identified."
        )

    with st.expander("F1 Score"):
        st.write(
            "F1 Score balances precision and recall. It is useful when "
            "working with imbalanced datasets."
        )

    st.warning(
        "Credit card fraud datasets are highly imbalanced. "
        "Accuracy alone is not enough to evaluate a fraud detection model."
    )


elif page == "Dataset Explorer":
    st.title("Dataset Explorer")

    st.markdown(
        '<div class="page-subtitle">'
        'Explore the transaction dataset used by the project.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Total Features", f"{len(feature_names):,}")
    col3.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")

    st.divider()
    st.subheader("Transaction Records")

    row_limit = st.selectbox(
        "Number of records to display",
        [5, 10, 25, 50, 100],
    )

    show_fraud_only = st.checkbox("Show only fraudulent transactions")

    display_df = df

    if show_fraud_only:
        display_df = df[df["Class"] == 1]

    st.dataframe(
        display_df.head(row_limit),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Displaying up to {min(row_limit, len(display_df)):,} "
        f"of {len(display_df):,} matching records."
    )

    st.divider()
    st.subheader("Dataset Information")

    with st.expander("View column data types"):
        dtype_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
            }
        )

        st.dataframe(
            dtype_df,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("View missing values"):
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]

        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True,
        )


elif page == "About Project":
    st.title("About This Project")

    st.markdown(
        '<div class="page-subtitle">'
        'Understanding the purpose, technology, and workflow '
        'of the fraud detection application.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Project Overview")

    st.write(
        """
        Credit Card Fraud Detection is a machine learning project
        designed to classify credit card transactions as legitimate
        or potentially fraudulent.

        The project uses historical transaction data to train
        supervised classification algorithms.
        """
    )

    st.divider()
    st.subheader("Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown("### Python")
        st.write(
            "Used for data processing, model training, and predictions."
        )

    with tech2:
        st.markdown("### Scikit-learn")
        st.write(
            "Used to train and evaluate the classification algorithms."
        )

    with tech3:
        st.markdown("### Streamlit")
        st.write(
            "Used to build the interactive frontend using Python."
        )

    st.divider()
    st.subheader("Machine Learning Algorithms")

    st.write("**1. Logistic Regression**")
    st.write(
        "A classification algorithm that estimates the probability "
        "of a transaction belonging to a class."
    )

    st.write("**2. Decision Tree**")
    st.write(
        "A tree-based algorithm that makes predictions using "
        "a sequence of decision rules."
    )

    st.write("**3. Random Forest**")
    st.write(
        "An ensemble algorithm that combines multiple decision trees "
        "to generate predictions."
    )

    st.divider()
    st.subheader("Target Variable")

    st.write("**Class = 0:** Legitimate transaction")
    st.write("**Class = 1:** Fraudulent transaction")

    st.divider()
    st.subheader("Project Workflow")

    st.markdown(
        """
        1. Load the historical transaction dataset.
        2. Prepare the input features.
        3. Split the data into training and testing sets.
        4. Train the classification algorithms.
        5. Evaluate their performance.
        6. Select a model using the evaluation results.
        7. Use the trained model to predict new input transactions.
        """
    )

    st.warning(
        "This project is for educational and demonstration purposes. "
        "Predictions should not be treated as definitive financial decisions."
    )

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:12px; color:#62748E;">
        <strong style="color:#2563EB;">FraudGuard</strong>
        &nbsp; | &nbsp; Credit Card Fraud Detection
        <br>
        <small>Machine Learning Project built with Python and Streamlit</small>
    </div>
    """,
    unsafe_allow_html=True,
)