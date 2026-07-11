import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")

model = joblib.load("fraud_model.pkl")
BEST_THRESHOLD = 0.997  # replace with your actual tuned threshold

if "history" not in st.session_state:
    st.session_state.history = []

def predict_df(df):
    df = df.copy()
    df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    probs = model.predict_proba(df)[:, 1]
    df["fraud_probability"] = probs
    df["prediction"] = (probs >= BEST_THRESHOLD).astype(int)
    return df

st.title("🛡️ Fraud Detection System")
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Batch Prediction", "📜 History"])

# ---------------- TAB 1: Single Prediction ----------------
with tab1:
    st.markdown("Enter transaction details to check for fraud risk.")
    col1, col2 = st.columns(2)

    with col1:
        transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
        amt = st.number_input("Amount", min_value=0.0, value=1000.0)
        oldbalanceOrg = st.number_input("Old Balance [Sender]", min_value=0.0, value=10000.0)

    with col2:
        newbalanceOrg = st.number_input("New Balance [Sender]", min_value=0.0, value=9000.0)
        oldbalanceDest = st.number_input("Old Balance [Receiver]", min_value=0.0, value=10000.0)
        newbalanceDest = st.number_input("New Balance [Receiver]", min_value=0.0, value=10000.0)

    # Validation
    errors = []
    if newbalanceOrg > oldbalanceOrg and transaction_type in ["TRANSFER", "CASH_OUT", "DEBIT"]:
        errors.append("Sender's new balance shouldn't exceed old balance for an outgoing transaction.")
    if amt == 0:
        errors.append("Amount should be greater than 0.")

    if errors:
        for e in errors:
            st.warning(e)

    if st.button("🔍 Predict", use_container_width=True, disabled=bool(errors)):
        input_data = pd.DataFrame([{
            "type": transaction_type, "amount": amt,
            "oldbalanceOrg": oldbalanceOrg, "newbalanceOrig": newbalanceOrg,
            "oldbalanceDest": oldbalanceDest, "newbalanceDest": newbalanceDest,
        }])
        result = predict_df(input_data)
        prob = result["fraud_probability"].iloc[0]
        is_fraud = result["prediction"].iloc[0] == 1

        st.subheader("Result")
        m1, m2 = st.columns(2)
        m1.metric("Fraud Probability", f"{prob*100:.2f}%")
        m2.metric("Threshold", f"{BEST_THRESHOLD*100:.2f}%")
        st.progress(min(float(prob), 1.0))

        if is_fraud:
            st.error("⚠️ This transaction is flagged as **FRAUDULENT**")
        else:
            st.success("✅ This transaction appears **LEGITIMATE**")

        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": transaction_type, "amount": amt,
            "fraud_probability": round(prob, 4),
            "prediction": "Fraud" if is_fraud else "Legit"
        })

        with st.expander("See transaction details sent to model"):
            st.dataframe(input_data)

# ---------------- TAB 2: Batch Prediction ----------------
with tab2:
    st.markdown("Upload a CSV with columns: `type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest`")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        required_cols = {"type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}

        if not required_cols.issubset(batch_df.columns):
            st.error(f"CSV missing required columns: {required_cols - set(batch_df.columns)}")
        else:
            result_df = predict_df(batch_df)
            n_fraud = (result_df["prediction"] == 1).sum()

            st.success(f"Processed {len(result_df)} transactions — {n_fraud} flagged as fraud")
            st.dataframe(result_df, use_container_width=True)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv, "fraud_predictions.csv", "text/csv")

# ---------------- TAB 3: History ----------------
with tab3:
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions made yet this session.")
