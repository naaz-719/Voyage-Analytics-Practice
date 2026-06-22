import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Voyage Analytics",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Voyage Analytics")
st.subheader("Integrating MLOps in Travel Productionization of ML Systems")

# ==========================
# LOAD MODELS
# ==========================

flight_model = joblib.load("models/flight_price_model.pkl")

# Load recommendation assets
user_similarity_df = joblib.load("models/user_similarity.pkl")

hotel_df = pd.read_csv("data/hotels.csv")

# ==========================
# SIDEBAR
# ==========================

page = st.sidebar.selectbox(
    "Select Module",
    [
        "Home",
        "Flight Price Prediction",
        "Gender Prediction",
        "Hotel Recommendation"
    ]
)

# ==========================
# HOME
# ==========================

if page == "Home":

    st.header("Project Overview")

    st.markdown("""
    ### Modules Included

    - Flight Price Prediction (Regression)
    - Gender Classification (Classification)
    - Hotel Recommendation System
    - MLflow Experiment Tracking
    - Flask API
    - Docker Containerization
    - Jenkins CI/CD Pipeline
    """)

# ==========================
# FLIGHT PRICE PREDICTION
# ==========================

elif page == "Flight Price Prediction":

    st.header("Flight Price Prediction")

    agency = st.number_input("Agency Encoded Value")

    flight_type = st.number_input("Flight Type Encoded Value")

    source = st.number_input("Source Encoded Value")

    destination = st.number_input("Destination Encoded Value")

    time = st.number_input("Travel Time")

    distance = st.number_input("Distance")

    if st.button("Predict Flight Price"):

        input_df = pd.DataFrame([{
            "agency": agency,
            "flightType": flight_type,
            "from": source,
            "to": destination,
            "time": time,
            "distance": distance
        }])

        prediction = flight_model.predict(input_df)

        st.success(
            f"Predicted Flight Price: {prediction[0]:.2f}"
        )

# ==========================
# GENDER CLASSIFICATION
# ==========================

elif page == "Gender Prediction":

    st.header("Gender Prediction")

    st.info(
        "Connect your deployed gender model here."
    )

    name = st.text_input("Enter Name")

    if st.button("Predict Gender"):

        st.success(
            "Connect Flask API endpoint result here."
        )

# ==========================
# HOTEL RECOMMENDATION
# ==========================

elif page == "Hotel Recommendation":

    st.header("Hotel Recommendation")

    user_id = st.number_input(
        "User ID",
        step=1
    )

    city = st.text_input("Preferred City")

    min_price = st.number_input(
        "Minimum Budget",
        value=0
    )

    max_price = st.number_input(
        "Maximum Budget",
        value=5000
    )

    if st.button("Recommend Hotels"):

        recommendations = hotel_df.copy()

        if city:
            recommendations = recommendations[
                recommendations["place"]
                .str.contains(city, case=False, na=False)
            ]

        recommendations = recommendations[
            (recommendations["hotel_price"] >= min_price)
            &
            (recommendations["hotel_price"] <= max_price)
        ]

        st.dataframe(
            recommendations[
                [
                    "hotel_name",
                    "place",
                    "hotel_price",
                    "days"
                ]
            ].head(10)
        )
