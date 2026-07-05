# Fraud Detection System

A Machine Learning-powered web application that predicts whether a financial transaction is fraudulent or legitimate. The project is built using Python, Scikit-learn, and Streamlit, providing an interactive interface for real-time fraud prediction.

---

## Features

- Predicts fraudulent transactions in real time
- Clean and interactive Streamlit interface
- Machine Learning pipeline for preprocessing and prediction
- Handles categorical and numerical features
- Fast and lightweight deployment

---

## Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Joblib

---

## Dataset Features

The model uses the following transaction details:

- Transaction Type
- Amount
- Old Balance (Origin)
- New Balance (Origin)
- Old Balance (Destination)
- New Balance (Destination)

---

## Project Structure

```
Fraud-Detection-System/
│
├── Fraud_Detector.py        # Streamlit application
├── fraud_model.pkl          # Trained ML model
├── README.md
└── EDA.ipynb           
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/Fraud-Detection-System.git
```

```bash
cd Fraud-Detection-System
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run Fraud_Detector.py
```

---

## How It Works

1. Enter the transaction details.
2. Click **Predict**.
3. The trained model analyzes the input.
4. The application displays whether the transaction is **Fraudulent** or **Legitimate**.

---

## Machine Learning Pipeline

- Data Preprocessing
- Handling Categorical Variables
- Feature Transformation
- Model Training
- Prediction

---

## Future Improvements

- Probability score for predictions
- Interactive dashboards
- Model comparison
- Fraud analytics
- Cloud deployment
- User authentication
- Transaction history

---

## Screenshots

<img width="607" height="305" alt="Screenshot 2026-07-06 045809" src="https://github.com/user-attachments/assets/7e930343-aa4b-4026-8e44-39b688876adb" />
<img width="624" height="200" alt="Screenshot 2026-07-06 045750" src="https://github.com/user-attachments/assets/1fea872f-5282-4bb2-8357-618d1c5634e4" />
<img width="642" height="344" alt="Screenshot 2026-07-06 045742" src="https://github.com/user-attachments/assets/a62eee9c-5bf5-496e-b5ae-237ebf6e563e" />



---

## Author

**Darshan**
