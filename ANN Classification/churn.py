import streamlit as st
import numpy as np 
import pandas as pd
import pickle 
import tensorflow as tf

# Load model and encoders
model = tf.keras.models.load_model('model.h5')

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("one_hot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)   

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)     

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn based on their demographic and account information.")

# Inputs
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 100)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure (Years)", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

if st.button("Predict"):
    # Encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))

    # Encode Gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Create input data
    input_dict = {
        "CreditScore": credit_score,
        "Gender": gender_encoded,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    }

    input_df = pd.DataFrame([input_dict])  # 1 row DataFrame

    # Combine with geo features
    input_final = pd.concat([input_df, geo_encoded_df], axis=1)

    # Scale the input
    input_scaled = scaler.transform(input_final)

    # Predict
    prediction = model.predict(input_scaled)

    if prediction[0][0] > 0.5:
        st.success("The customer is likely to churn.")
    else:
        st.success("The customer is unlikely to churn.")
    st.write("Prediction Probability:", prediction[0][0])
