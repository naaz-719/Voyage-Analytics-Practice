import streamlit as st
import pandas as pd
import requests

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="Voyage Analytics",
    page_icon="✈️",
    layout="wide"
)

API_URL = "https://voyage-analytics-practice-2.onrender.com"

# Load flight dataset
flights_df = pd.read_csv("data/flights.csv")

# Session state
if "destination_city" not in st.session_state:
    st.session_state.destination_city = ""

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("✈️ Voyage Analytics")

page = st.sidebar.selectbox(
    "Select Module",
    [
        "Flight Price Prediction",
        "Gender Prediction",
        "Hotel Recommendation"
    ]
)

# ==================================================
# FLIGHT PRICE PREDICTION
# ==================================================

if page == "Flight Price Prediction":

    st.title("✈️ Flight Price Prediction")
    st.markdown("Predict airline ticket prices using Machine Learning")

    try:

        options = requests.get(
            f"{API_URL}/flight-options"
        ).json()

        col1, col2 = st.columns(2)

        with col1:
            from_city = st.selectbox(
                "From City",
                options["from"]
            )

        with col2:

            to_options = [
                city
                for city in options["to"]
                if city != from_city
            ]

            to_city = st.selectbox(
                "To City",
                to_options
            )

        # Save destination for hotel page
        st.session_state.destination_city = to_city

        # Find route info automatically
        route = flights_df[
            (flights_df["from"] == from_city)
            &
            (flights_df["to"] == to_city)
        ]

        if not route.empty:

            distance = float(
                route.iloc[0]["distance"]
            )

            time = float(
                route.iloc[0]["time"]
            )

        else:

            distance = 0
            time = 0

        st.markdown("### Route Information")

        col3, col4 = st.columns(2)

        with col3:
            st.metric(
                "Distance (KM)",
                f"{distance:.2f}"
            )

        with col4:
            st.metric(
                "Duration (Hours)",
                f"{time:.2f}"
            )

        col5, col6 = st.columns(2)

        with col5:
            flight_type = st.selectbox(
                "Flight Type",
                options["flightType"]
            )

        with col6:
            agency = st.selectbox(
                "Agency",
                options["agency"]
            )

        if st.button(
            "Predict Flight Price",
            use_container_width=True
        ):

            payload = {
                "from": from_city,
                "to": to_city,
                "flightType": flight_type,
                "agency": agency,
                "time": time,
                "distance": distance
            }

            response = requests.post(
                f"{API_URL}/predict-flight",
                json=payload
            )

            result = response.json()

            if "predicted_price" in result:

                st.success(
                    "Prediction Complete"
                )

                st.metric(
                    "Predicted Price",
                    f"${result['predicted_price']:.2f}"
                )

            else:

                st.error(result)

    except Exception as e:

        st.error(str(e))

# ==================================================
# GENDER PREDICTION
# ==================================================

elif page == "Gender Prediction":

    st.title("👤 Gender Prediction")
    st.markdown(
        "Predict gender using Machine Learning"
    )

    name = st.text_input(
        "Enter Name"
    )

    if st.button(
        "Predict Gender",
        use_container_width=True
    ):

        if name:

            response = requests.post(
                f"{API_URL}/predict-gender",
                json={
                    "name": name
                }
            )

            result = response.json()

            if "predicted_gender" in result:

                st.success(
                    f"Predicted Gender: {result['predicted_gender']}"
                )

            else:

                st.error(result)

# ==================================================
# HOTEL RECOMMENDATION
# ==================================================

elif page == "Hotel Recommendation":

    st.title("🏨 Hotel Recommendation")

    city = st.session_state.destination_city

    if city:

        st.info(
            f"Showing hotels in: {city}"
        )

    else:

        st.warning(
            "Please select a destination city from Flight Prediction first."
        )

    max_price = st.number_input(
        "Maximum Budget",
        min_value=0,
        value=500
    )

    if st.button(
        "Find Hotels",
        use_container_width=True
    ):

        payload = {
            "city": city,
            "min_price": 0,
            "max_price": max_price
        }

        response = requests.post(
            f"{API_URL}/recommend-hotels",
            json=payload
        )

        hotels = response.json()

        if isinstance(hotels, list) and len(hotels) > 0:

            st.success(
                f"{len(hotels)} Hotels Found"
            )

            df = pd.DataFrame(
                hotels
            )

            st.dataframe(
                df,
                use_container_width=True
            )

        else:

            st.warning(
                "No hotels found for this city and budget."
            )

# ==================================================
# FOOTER
# ==================================================

st.sidebar.markdown("---")
st.sidebar.success(
    "Flask API + Streamlit + Machine Learning"
)
