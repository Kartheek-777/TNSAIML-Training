# Credit Card Fraud Detection Using Classification Techniques

## Project Overview

Credit Card Fraud Detection is a Machine Learning project that identifies fraudulent credit card transactions using classification techniques. The system analyzes transaction data and predicts whether a transaction is legitimate or fraudulent.

This project uses three classification algorithms:
- Logistic Regression
- Decision Tree
- Random Forest

The best-performing model is selected based on evaluation metrics and saved for future predictions. A Streamlit web application provides an interactive interface to test transactions and view model performance.

## Objectives

- Detect fraudulent credit card transactions.
- Apply different machine learning classification algorithms.
- Compare model performance using evaluation metrics.
- Identify the best-performing classification model.
- Build an interactive web application using Streamlit.

## Technologies Used

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Joblib
- **Frontend:** Streamlit
- **Visualization:** Matplotlib
- **Dataset:** Kaggle Credit Card Fraud Detection Dataset

## Machine Learning Algorithms

1. **Logistic Regression:** Classifies transactions into legitimate or fraudulent categories.
2. **Decision Tree:** Uses decision-making rules to classify transactions.
3. **Random Forest:** Combines multiple decision trees to improve classification performance.

## Project Structure

```text
CreditCardFraudDetection/
│
├── dataset/
│   └── creditcard.csv
│
├── train.py
├── app.py
├── model.joblib
├── model_results.csv
├── requirements.txt
└── README.md
```

## Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle.

- `0` – Legitimate Transaction
- `1` – Fraudulent Transaction

The dataset contains transaction details, including Time, Amount, and anonymized features (V1 to V28).

**Dataset Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Kartheek-777/TNSAIML-Training.git
```

### 2. Navigate to the Project Folder

```bash
cd CreditCardFraudDetection
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 6. Add the Dataset

Download the dataset from Kaggle and place `creditcard.csv` inside the `dataset` folder.

### 7. Train the Machine Learning Models

```bash
python train.py
```

This step trains the classification models, compares their performance, and saves the best model as `model.joblib`.

### 8. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## Model Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy:** Measures the overall percentage of correct predictions.
- **Precision:** Measures how many predicted fraud transactions are actually fraudulent.
- **Recall:** Measures how many actual fraud transactions are correctly identified.
- **F1 Score:** Balances precision and recall.
- **Confusion Matrix:** Shows correct and incorrect classifications.

## Application Features

- View total transaction statistics.
- Display legitimate and fraudulent transaction counts.
- Enter transaction details for fraud prediction.
- Predict whether a transaction is legitimate or fraudulent.
- Display fraud prediction probability when available.
- Compare classification model performance.
- Explore dataset records.
- Download prediction results.

## Output Files

- `model.joblib` – Stores the selected trained machine learning model and feature information.
- `model_results.csv` – Stores the performance comparison of classification models.

## Future Enhancements

- Integrate real-time transaction monitoring.
- Improve model performance using advanced classification techniques.
- Deploy the application to a cloud platform.
- Add transaction history and fraud alerts.

## Author

**Kartheek Lagisetti**

GitHub: [Kartheek-777](https://github.com/Kartheek-777)

## Conclusion

This project demonstrates how machine learning classification techniques can be used to identify fraudulent credit card transactions. By comparing Logistic Regression, Decision Tree, and Random Forest, the system selects a suitable model and provides an interactive fraud detection application using Streamlit.
