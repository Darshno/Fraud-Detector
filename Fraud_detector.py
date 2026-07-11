import streamlit as st 
import pandas as pd
import joblib

model = joblib.load("fraud_model.pkl")

st.title("Fraud Detection Prediction app")

st.markdown("Please enter the Transaction Details")

st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT","TRANSFER","CASH OUT","DEPOSIT"])
amt = st.number_input(
    "Amount",
    min_value=0.0,   # float
    value=1000.0       # int
)
oldbalanceOrg = st.number_input("Old Ballance [Sender]", min_value = 0.0 , value = 10000.0)
newbalanceOrg = st.number_input("New Ballance [Sender]", min_value = 0.0 , value = 9000.0)
oldbalanceDest = st.number_input("Old Ballance [Reciever]", min_value = 0.0 , value = 10000.0)
newbalanceDest = st.number_input("New Ballance [Reciever]", min_value = 0.0 , value = 10000.0)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amt,
        "oldbalanceOrg" : oldbalanceOrg,
        "newbalanceOrig" : newbalanceOrg,
        "oldbalanceDest" : oldbalanceDest,
        "newbalanceDest" : newbalanceDest
    }])

    prediction = model.predict(input_data)

    st.subheader(f"Prediction: {prediction[0]}")

    if prediction[0] == 1:
        st.error("this Transaction is Fraud")
    else:
        st.success("This Transaction is not Fraud")
