# Fraud Detection System
A Machine Learning-powered web application that predicts whether a financial transaction is fraudulent or legitimate. Built using Python, XGBoost, Scikit-learn, and Streamlit, with a full data science pipeline covering feature engineering, class imbalance handling, threshold optimization, and cross-validated evaluation.

---

## Results

| Metric | Score |
|---|---|
| F1-score (5-fold CV) | **0.94 ± 0.004** |
| Precision (tuned threshold) | 0.94 |
| Recall (tuned threshold) | 0.94 |
| ROC-AUC | *(add your value)* |
| Dataset size | 2M+ transactions, 0.13% fraud rate |

Model: **XGBoost Classifier** with `scale_pos_weight` for class imbalance and precision-recall threshold optimization (default 0.5 threshold performed poorly — tuned to ~0.997 for optimal F1).

---

## Visualizations

**Class Distribution**
<img width="900" height="600" alt="cv_f1_scores" src="https://github.com/user-attachments/assets/d62f0562-2ef5-48db-81ae-dae053dcc296" />

**Feature Correlation**
<img width="1050" height="900" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/9a721e9c-42f0-4edc-a2aa-c1890c2672e9" />

**Feature Importance**
<img width="960" height="720" alt="feature_importance" src="https://github.com/user-attachments/assets/59df7636-3353-4115-b6a9-e8d461c075a0" />

**Precision-Recall Curve**
<img width="900" height="750" alt="precision_recall_curve" src="https://github.com/user-attachments/assets/5ca4c5f7-8ffd-4a27-9929-1b811e11ee39" />

**ROC Curve**
<img width="900" height="750" alt="roc_curve" src="https://github.com/user-attachments/assets/49953caf-e164-47e1-89f2-7546bf750eb0" />

**Confusion Matrix (Tuned Threshold)**
<img width="750" height="750" alt="confusion_matrix" src="https://github.com/user-attachments/assets/5f53ac1c-3c07-47e6-b63e-7f6ac2c662fa" />


---

## Features
- Predicts fraudulent transactions in real time
- Clean and interactive Streamlit interface
- End-to-end ML pipeline: preprocessing, feature engineering, training, threshold-tuned prediction
- Handles severe class imbalance (0.13% fraud rate) via `scale_pos_weight`
- Engineered balance-discrepancy features to boost fraud signal detection

---

## Tech Stack
- Python
- XGBoost
- Scikit-learn
- Pandas / NumPy
- Streamlit
- Joblib

---

## Dataset Features
- Transaction Type
- Amount
- Old Balance (Origin) / New Balance (Origin)
- Old Balance (Destination) / New Balance (Destination)
- **Engineered:** `errorBalanceOrig` = oldbalanceOrg − amount − newbalanceOrig
- **Engineered:** `errorBalanceDest` = oldbalanceDest + amount − newbalanceDest

---

## Project Structure
Fraud-Detection-System/
│
├── Fraud_Detector.py        # Streamlit application
├── fraud_model.pkl          # Trained XGBoost pipeline
├── README.md
└── EDA.ipynb                # Full training, tuning & CV notebook

---

## Installation
```bash
git clone https://github.com/yourusername/Fraud-Detection-System.git
cd Fraud-Detection-System
pip install -r requirements.txt
streamlit run Fraud_Detector.py
```

---

## How It Works
1. Enter transaction details.
2. Click **Predict**.
3. The trained XGBoost pipeline computes a fraud probability.
4. The app applies the optimized threshold (~0.997) and displays **Fraudulent** or **Legitimate**.

---

## Machine Learning Pipeline
1. Data preprocessing (scaling, one-hot encoding)
2. Feature engineering (balance-discrepancy features)
3. Class imbalance handling (`scale_pos_weight`)
4. XGBoost model training
5. Precision-recall threshold optimization
6. 5-fold stratified cross-validation for robustness
7. Feature importance analysis (leakage check)

---

## Future Improvements
- SHAP explainability for individual predictions
- Interactive analytics dashboard
- Model comparison (LightGBM, CatBoost)
- REST API deployment
- Cloud hosting
- User authentication & transaction history

---

## Screenshots
<img width="607" height="305" alt="Screenshot 2026-07-06 045809" src="https://github.com/user-attachments/assets/7e930343-aa4b-4026-8e44-39b688876adb" />
<img width="624" height="200" alt="Screenshot 2026-07-06 045750" src="https://github.com/user-attachments/assets/1fea872f-5282-4bb2-8357-618d1c5634e4" />
<img width="642" height="344" alt="Screenshot 2026-07-06 045742" src="https://github.com/user-attachments/assets/a62eee9c-5bf5-496e-b5ae-237ebf6e563e" />


---

## Author
**Darshan**
