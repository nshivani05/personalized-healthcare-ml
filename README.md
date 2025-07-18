# Personalized Healthcare Recommendations – A Machine Learning Approach

This project builds a machine learning model to predict whether an individual is likely to donate blood again, based on historical donation behavior. The system acts like a healthcare recommendation engine by identifying high-risk drop-offs and providing insights to re-engage potential donors.

---
## Live Demo

👉 [Click here to try the live app](https://personalized-healthcare-ml-asu5dt8necs7eewqjuuxpk.streamlit.app/)


## Problem Statement

The objective is to use past donation data to:
- Predict future blood donation behavior
- Recommend personalized engagement strategies
- Help healthcare organizations retain consistent donors

---

## Dataset Overview

- **Source:** Public datasets, surveys, and donor records
- **Entries:** 548
- **Features:** Recency, Frequency, Monetary, Time
- **Target:** `Class` (1 = will donate again, 0 = will not)

---

## Exploratory Data Analysis (EDA)

- ~75% of users did not donate again (imbalanced dataset)
- Frequency and Monetary had strong positive correlation
- Recent, frequent donors were more likely to return
<img width="814" height="409" alt="image" src="https://github.com/user-attachments/assets/1852c38e-0d71-4566-9f9a-20ff04ea3a16" />

<img width="633" height="451" alt="image" src="https://github.com/user-attachments/assets/7030484c-d9fa-4c47-87a9-8d925d7d8662" />

---

## Data Preprocessing

- Handled missing values
- Applied **StandardScaler** for normalization
- Used **Stratified Train-Test Split** (80/20) to maintain class distribution

---

## Model Training

- **Model:** Logistic Regression (Scikit-learn)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score
- **Result:** ~77% Accuracy, balanced metrics

📌 *Classification logic:*
- `Class 1` → Likely to donate again
- `Class 0` → Unlikely to donate again

---

## Recommendation Logic

Used model predictions to simulate a recommendation engine:
- Engage likely donors with thank-you emails
- Follow up with lower-likelihood donors via educational outreach
- Prioritize frequent contributors for future drives

---
## 💡 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
## Results Snapshot

<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/295acb00-9b36-46c1-ab39-a740a9b74d93" />


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Example model and evaluation (pseudocode)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))


