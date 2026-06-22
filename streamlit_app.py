import streamlit as st
import pandas as pd
import requests

st.set_page_config(
page_title="Voyage Analytics",
page_icon="✈️",
layout="wide"
)

API_URL = "https://voyage-analytics-practice-2.onrender.com"

st.sidebar.title("✈️ Voyage Analytics")

page = st.sidebar.radio(
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

```
st.title("✈️ Flight Price Prediction")

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
            city for city in options["to"]
            if city != from_city
        ]

        to_city = st.selectbox(
            "To City",
            to_options
        )

    col3, col4 = st.columns(2)

    with col3:
        flight_type = st.selectbox(
            "Flight Type",
            options["flightType"]
        )

    with col4:
        agency = st.selectbox(
            "Agency",
            options["agency"]
        )

    col5, col6 = st.columns(2)

    with col5:
        time = st.number_input(
            "Flight Duration (Hours)",
            min_value=0.0,
            value=1.5
        )

    with col6:
        distance = st.number_input(
            "Distance (KM)",
            min_value=0,
            value=1000
        )

    if st.button("Predict Flight Price"):

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

            st.success("Prediction Complete!")

            st.metric(
                "Predicted Flight Price",
                f"${result['predicted_price']}"
            )

        else:
            st.error(result)

except Exception as e:
    st.error(str(e))
```

# ==================================================

# GENDER PREDICTION

# ==================================================

elif page == "Gender Prediction":

```
st.title("👤 Gender Prediction")

name = st.text_input(
    "Enter Name",
    placeholder="Naaz"
)

if st.button("Predict Gender"):

    response = requests.post(
        f"{API_URL}/predict-gender",
        json={"name": name}
    )

    result = response.json()

    if "predicted_gender" in result:

        st.success(
            f"Predicted Gender: {result['predicted_gender']}"
        )

        st.metric(
            "Prediction",
            result["predicted_gender"]
        )

    else:
        st.error(result)
```

# ==================================================

# HOTEL RECOMMENDATION

# ==================================================

elif page == "Hotel Recommendation":

```
st.title("🏨 Hotel Recommendation")

city = st.text_input(
    "City",
    placeholder="Salvador (BH)"
)

max_price = st.number_input(
    "Maximum Price",
    min_value=0,
    value=500
)

if st.button("Find Hotels"):

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

        df = pd.DataFrame(hotels)

        st.dataframe(
            df,
            use_container_width=True
        )

    else:
        st.warning(
            "No hotels found."
        )
```

st.sidebar.markdown("---")
st.sidebar.info(
"Built with Flask + Streamlit + Machine Learning"
)

