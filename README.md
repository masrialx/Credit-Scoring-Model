# Credit Scoring Model

## Introduction

Bati Bank has partnered with an eCommerce company to introduce a buy-now-pay-later service. This project aims to develop a Credit Scoring Model to assess a customer's creditworthiness and predict the likelihood of default.

### Project Objectives:
- Define a proxy variable to categorize users as high-risk (bad) or low-risk (good).
- Select important features correlated with default probability.
- Develop a model that assigns risk probability to new customers.
- Develop a model to predict the optimal loan amount and duration.
- Deploy a REST API to serve the credit scoring model.

---

## Setup Instructions

To set up the project, follow these steps:

### 1. Clone the Repository
Start by cloning the repository to your local machine or environment:

```bash
git clone https://github.com/yourusername/credit-scoring-model.git
cd credit-scoring-model
```

### 2. Install Dependencies

Create a virtual environment to keep your dependencies isolated:

```bash
# For Linux/MacOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include the following libraries:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
flask
awscli
```

### 3. Prepare the Data

Ensure that you have the dataset stored locally or on Google Drive (if working in Google Colab). The dataset should be in CSV format and have the following columns:
- Customer & Transaction Information: `TransactionId`, `CustomerId`, `TransactionStartTime`, `Amount`, `CurrencyCode`.
- Product Information: `ProductId`, `ProductCategory`.
- Fraud Detection: `FraudResult`.

Modify the data path in your scripts as required.

### 4. Train the Model

Run the Jupyter notebook or Python script to train the model. The training script will handle:
- Data loading and preprocessing
- Feature engineering
- Model training (Logistic Regression, Decision Trees, Random Forest, XGBoost)
- Hyperparameter tuning using GridSearch or RandomSearch

```bash
python train_model.py
```

The trained model will be saved as a serialized file (e.g., `model.pkl`).

---

## Data Exploration (EDA)

The dataset includes customer transaction data with the following fields:
- **Customer & Transaction Information**: `TransactionId`, `CustomerId`, `TransactionStartTime`, `Amount`, `CurrencyCode`.
- **Product Information**: `ProductId`, `ProductCategory`.
- **Fraud Detection**: `FraudResult` (1 for fraud, 0 for no fraud).

### Key Steps:
1. **Summary Statistics**: Analyze mean, standard deviation, min, max for key features (`Amount`, `Transaction Count`).
2. **Missing Values**: Handle missing data using imputation methods.
3. **Outlier Detection**: Identify and cap extreme values in `Amount` using Winsorization.
4. **Feature Correlation**: Use heatmap to visualize correlations, especially between `Amount`, `TransactionCount`, and fraud risk.

---

## Feature Engineering

1. **Aggregated Features**:  
   - Total Transaction Amount per customer  
   - Average Transaction Amount  
   - Transaction Count per customer  
   - Standard Deviation of Transaction Amounts  

2. **Extracted Features**:  
   - Time-based features: Hour, Day, Month, Year

3. **Encoding Categorical Variables**:  
   - One-Hot Encoding for categorical features  
   - Label Encoding for ordinal data

4. **Normalization and Standardization**:  
   - Standardization applied to numerical features.

5. **Weight of Evidence (WoE)**:  
   - Used for binning and analyzing customer behavior in transactions.

---

## Default Estimator

### RFMS Segmentation:
- **Recency**: Days since last transaction
- **Frequency**: Total transaction count
- **Monetary**: Average transaction value
- **Seasonality**: Active periods of the customer

This segmentation classifies customers into good or bad risk categories based on transaction patterns.

---

## Machine Learning Models

The following models are trained and evaluated:
- **Logistic Regression**
- **Decision Trees**
- **Random Forest**
- **XGBoost**

### Model Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

### Model Performance:
| Model           | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-----------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 85%     | 80%       | 78%    | 79%      | 0.88    |
| Random Forest   | 90%     | 85%       | 87%    | 86%      | 0.92    |
| XGBoost         | 92%     | 88%       | 89%    | 88%      | 0.94    |

### Hyperparameter Tuning:
- Used Grid Search and Random Search to optimize model performance.

---

## Model Deployment (REST API)

### Framework:
- Flask was used to build the API for serving the model.

### API Endpoint:
- `predict_credit_score()`: Predicts credit risk and loan amount based on customer transaction details.

### API Request:
- **Input**: JSON containing customer transaction details.
- **Output**: Credit risk probability and recommended loan amount.

### Deployment:
- Deployed the model on **AWS Lambda** with **API Gateway** to handle requests.

---

## Conclusion & Future Work

### Key Insights:
- High-risk customers typically have frequent transactions but lower amounts.
- Fraudulent transactions occur mostly at night.
- **XGBoost** provides the best performance in predicting credit risk.

### Future Enhancements:
- Integrate real-time data streaming.
- Explore deep learning models for improved predictions.
- Use alternative data sources (e.g., social media behavior) to enhance credit scoring.

---

## Appendices

- **Code snippets**: Key analysis steps and functions.
- **Feature Descriptions**: Detailed explanations of each feature.
- **Visualizations**: EDA charts and model performance plots.