import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load model and preprocessing tools
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# --- Streamlit Page Config ---
st.set_page_config(page_title="ChurnGuard", layout="centered")

# --- Page Navigation ---
if "page" not in st.session_state:
    st.session_state.page = "intro"

# --- Page 1: Introduction ---
if st.session_state.page == "intro":
    st.title("🛡️ ChurnGuard")

    st.markdown("""
    Welcome to **ChurnGuard** – your smart assistant to predict customer churn.

    📊 **What it does**:  
    This app predicts whether a customer is likely to leave your bank based on their profile and behavior.

    ✅ Easy to use  
    ✅ Instant insights

    Click the button below to begin the churn analysis.
    """)

    if st.button("🔍 Start Churn Prediction"):
        st.session_state.page = "predict"
        st.rerun()

# --- Page 2: Churn Prediction ---
elif st.session_state.page == "predict":
    st.title("🔍 Customer Churn Prediction")

    # Input form
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 30)
    balance = st.number_input('💰 Balance', min_value=0.0)
    credit_score = st.number_input('📊 Credit Score', min_value=300, max_value=900, value=600)
    estimated_salary = st.number_input('🏦 Estimated Salary', min_value=0.0)
    tenure = st.slider('📅 Tenure (Years with Bank)', 0, 10, 5)
    num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('💳 Has Credit Card?', ['No', 'Yes'])
    is_active_member = st.selectbox('📈 Is Active Member?', ['No', 'Yes'])

    # Prediction button
    if st.button("📤 Predict Churn"):
        has_cr_card = 1 if has_cr_card == 'Yes' else 0
        is_active_member = 1 if is_active_member == 'Yes' else 0

        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        input_data_scaled = scaler.transform(input_data)

        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        st.subheader("📢 Prediction Result:")
        st.progress(int(prediction_proba * 100))
        st.write(f"**Churn Probability:** `{prediction_proba:.2%}`")

        if prediction_proba > 0.5:
            st.error("⚠️ The customer is **likely to churn**.")
        else:
            st.success("✅ The customer is **not likely to churn**.")

    if st.button("🔙 Back to Home"):
        st.session_state.page = "intro"
        st.rerun()