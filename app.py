
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load trained model and encoders
# -----------------------------

@st.cache_resource
def load_artifacts():
    model = load_model("model.h5", compile=False)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("one_hot_encoder_geography.pkl", "rb") as f:
        one_hot_encoder_geography = pickle.load(f)

    return model, scaler, label_encoder_gender, one_hot_encoder_geography


model, scaler, label_encoder_gender, one_hot_encoder_geography = load_artifacts()


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer will leave the bank.")

# Customer inputs
geography = st.selectbox(
    "Geography",
    one_hot_encoder_geography.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider("Age", 18, 100, 30)

balance = st.number_input(
    "Account Balance",
    min_value=0.0,
    value=50000.0
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

tenure = st.slider("Tenure", 0, 10, 5)

num_of_products = st.slider(
    "Number of Products",
    1, 4, 1
)

has_cr_card = st.selectbox(
    "Has Credit Card?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

is_active_member = st.selectbox(
    "Is Active Member?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Churn"):

    # Encode Gender using saved LabelEncoder
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # One-hot encode Geography using saved encoder
    geography_encoded = one_hot_encoder_geography.transform(
        pd.DataFrame({"Geography": [geography]})
    )

    # Convert sparse matrix to dense if needed
    if hasattr(geography_encoded, "toarray"):
        geography_encoded = geography_encoded.toarray()

    geography_columns = one_hot_encoder_geography.get_feature_names_out(
        ["Geography"]
    )

    geography_df = pd.DataFrame(
        geography_encoded,
        columns=geography_columns
    )

    # Create customer input DataFrame
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # Combine input with encoded geography
    input_data = pd.concat(
        [input_data, geography_df],
        axis=1
    )

    # Match training feature order
    if hasattr(scaler, "feature_names_in_"):
        input_data = input_data[
            list(scaler.feature_names_in_)
        ]
    else:
        st.error(
            "Scaler does not contain feature names. "
            "Verify the original training column order."
        )
        st.stop()

    # Scale input data using saved scaler
    input_scaled = scaler.transform(input_data)

    # Predict using trained ANN model
    prediction = model.predict(input_scaled, verbose=0)

    # Extract churn probability
    churn_probability = float(prediction[0][0])

    # Display result
    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{churn_probability * 100:.2f}%"
    )

    if churn_probability >= 0.5:
        st.error(
            "The customer is likely to leave the bank."
        )
    else:
        st.success(
            "The customer is likely to stay with the bank."
        )
