import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Loading Models
rf_model = joblib.load("models/random_forest_model.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

# Preprocessing
def preprocess_input(df):
    df_proc = df.copy()
    
    # Label encode categorical columns
    for col, le in label_encoders.items():
        if col in df_proc.columns:
            df_proc[col] = le.transform(df_proc[col])
    
    # Convert all columns to numeric (int/float) to avoid XGBoost dtype errors
    df_proc = df_proc.apply(pd.to_numeric)
    
    return df_proc


# Prediction Function
def predict_churn(model, df):
    preds = model.predict(df)
    probs = model.predict_proba(df)[:,1] if hasattr(model, "predict_proba") else None
    return preds, probs

# Streamlit Interface
st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict churn probability.")

# Model selection
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

# Columns for input features
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
    login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Computer"])
    city_tier = st.selectbox("City Tier", ["1", "2", "3"])
    warehouse_home = st.number_input("Warehouse to Home (distance)", min_value=0, max_value=1000, value=50)
    payment_mode = st.selectbox("Preferred Payment Mode", ["Debit Card", "UPI", "Credit Card", "Cash on Delivery", "E wallet"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    hour_spend = st.number_input("Hours Spent on App", min_value=0, max_value=24, value=2)
    num_devices = st.number_input("Number of Devices Registered", min_value=0, max_value=10, value=1)

with col2:
    order_cat = st.selectbox("Preferred Order Category", ["Laptop & Accessory", "Mobile Phone", "Others", "Fashion", "Grocery"])
    satisfaction_score = st.slider("Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
    marital_status = st.selectbox("Marital Status", ["Single", "Divorced", "Married"])
    num_address = st.number_input("Number of Addresses", min_value=1, max_value=10, value=1)
    complain = st.selectbox("Complain History", ["Yes", "No"])
    complain_numeric = 1 if complain == "Yes" else 0
    order_hike = st.number_input("Order Amount Hike from Last Year (%)", min_value=0, max_value=500, value=10)
    coupon_used = st.number_input("Coupons Used", min_value=0, max_value=50, value=5)
    order_count = st.number_input("Order Count", min_value=0, max_value=1000, value=10)
    days_since_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=365, value=30)
    cashback = st.number_input("Cashback Amount", min_value=0, max_value=10000, value=100)

# Collecting input
input_df = pd.DataFrame({
    "Tenure": [tenure],
    "PreferredLoginDevice": [login_device],
    "CityTier": [city_tier],
    "WarehouseToHome": [warehouse_home],
    "PreferredPaymentMode": [payment_mode],
    "Gender": [gender],
    "HourSpendOnApp": [hour_spend],
    "NumberOfDeviceRegistered": [num_devices],
    "PreferedOrderCat": [order_cat],
    "SatisfactionScore": [satisfaction_score],
    "MaritalStatus": [marital_status],
    "NumberOfAddress": [num_address],
    "Complain": [complain_numeric],
    "OrderAmountHikeFromlastYear": [order_hike],
    "CouponUsed": [coupon_used],
    "OrderCount": [order_count],
    "DaySinceLastOrder": [days_since_last_order],
    "CashbackAmount": [cashback]
})

# Prediction
if st.button("Predict Churn"):
    processed_input = preprocess_input(input_df)
    prediction, probability = predict_churn(model, processed_input)
    
    # Result Display
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.markdown("<span style='color:red; font-size:20px;'>Customer will churn!</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:green; font-size:20px;'>Customer will NOT churn!</span>", unsafe_allow_html=True)
    st.write(f"Churn Probability: {probability[0]:.2f}")

