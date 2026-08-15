# Bank Marketing Campaign Prediction

## 1. Problem Statement

The objective of this project is to predict whether a client will subscribe to a bank term deposit as a result of a bank marketing campaign.

The problem is formulated as a binary classification task where:

- `yes` = client subscribed to a term deposit
- `no` = client did not subscribe to a term deposit

The project compares five machine learning classification models and evaluates their performance using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

---

## 2. Dataset Description

The dataset used for this project is the **Bank Marketing dataset with additional social and economic context attributes**.

The dataset contains 41,188 customer records and 20 input attributes, along with the binary target variable `y`.

The input variables include customer demographic information, contact information, previous campaign information, campaign-related variables, and social/economic indicators.

Examples of input variables include:

- Age
- Job
- Marital status
- Education
- Credit default status
- Housing loan
- Personal loan
- Contact type
- Month
- Day of week
- Call duration
- Number of campaign contacts
- Previous campaign information
- Previous campaign outcome
- Employment variation rate
- Consumer price index
- Consumer confidence index
- Euribor 3-month rate
- Number of employees

The dataset contains categorical values labelled `unknown`. These were retained as a separate category rather than deleting the corresponding observations.

The target variable is:

- `y = yes`: client subscribed to a term deposit
- `y = no`: client did not subscribe

The dataset is publicly available and is based on the UCI Bank Marketing dataset.

---

## 3. GitHub Repository

GitHub Repository:

https://github.com/mehul22051999/ML-Assignment-2

The repository contains:

- Source code
- Jupyter notebook containing data exploration and model implementation
- `requirements.txt`
- `test_data.csv`
- Saved machine learning models
- Streamlit application

---

## 4. Models Used

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

### Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9012 | 0.8008 | 0.6900 | 0.2231 | 0.3371 | 0.3550 |
| Decision Tree | 0.9022 | 0.7909 | 0.6805 | 0.2478 | 0.3633 | 0.3715 |
| KNN | 0.8956 | 0.7341 | 0.5752 | 0.2802 | 0.3768 | 0.3525 |
| Naive Bayes | 0.7288 | 0.7729 | 0.2536 | 0.7241 | 0.3756 | 0.3070 |
| Random Forest | 0.9031 | 0.8082 | 0.7181 | 0.2306 | 0.3491 | 0.3710 |

---

## 5. Observations on Model Performance

### Logistic Regression

Logistic Regression achieved an accuracy of 90.12% and precision of 69.00%. Its AUC was 0.8008. However, its recall was relatively low at 22.31%, indicating that the model missed many customers who actually subscribed to the term deposit.

### Decision Tree

The Decision Tree achieved 90.22% accuracy and an MCC of 0.3715, which was the highest MCC among the five models. Its recall was 24.78%, so although its overall performance was strong, it still missed a significant number of positive cases.

### KNN

KNN achieved the highest F1 Score of 0.3768 among the five models. This indicates the best balance between precision and recall according to the F1 metric. However, its AUC of 0.7341 was the lowest among the models.

### Naive Bayes

Naive Bayes achieved the highest recall at 72.41%. This means it identified a much larger proportion of the customers who actually subscribed. However, its precision was only 25.36%, resulting in a large number of false positive predictions. Its overall accuracy was also substantially lower at 72.88%.

### Random Forest

Random Forest achieved the highest accuracy (90.31%), precision (71.81%), and AUC (0.8082). However, its recall was only 23.06%, meaning that it missed many of the actual positive cases despite performing well on the other metrics.

---

## 6. Overall Winner

Based on the F1 Score, **KNN is selected as the overall winner**, with an F1 Score of 0.3768.

F1 Score was considered important because the dataset is imbalanced and accuracy alone does not adequately represent the ability of a model to identify customers who subscribe to the term deposit.

However, the choice of model can depend on the business objective. If the bank prioritizes identifying as many potential customers as possible, Naive Bayes may be preferred because it achieved the highest recall of 72.41%. If precision and overall discrimination are more important, Random Forest provides the strongest combination of accuracy, precision, and AUC.

---

## 7. Streamlit Application

The trained models have been deployed through a Streamlit web application.

Streamlit App:

https://ml-assignment-2-mvjwitqnrch5gs4hjyvvyy.streamlit.app/

The application provides:

- CSV test-data upload
- Machine learning model selection
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
- Confusion matrix
- Classification report

Only test data is intended to be uploaded to the application.

---

## 8. Project Structure

```text
ML-Assignment-2/
│
├── app.py
├── exploration.ipynb
├── requirements.txt
├── test_data.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── scaler.pkl
    └── feature_columns.pkl
