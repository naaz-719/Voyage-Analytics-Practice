import streamlit as st
import pandas as pd
import requests

# ==================================================

# PAGE CONFIG

# ==================================================

st.set_page_config(
page_title="Voyage Analytics",
page_icon="✈️",
layout="wide"
)

API_URL = "https://voyage-analytics-practice-2.onrender.com"

# ==================================================

# SIDEBAR

# ==================================================

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

        with st.spinner(
            "Predicting..."
        ):

            response = requests.post(
                f"{API_URL}/predict-flight",
                json=payload
            )

            result = response.json()

            if "predicted_price" in result:

                st.success(
                    "Prediction Complete!"
                )

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
st.markdown(
    "Predict gender from first names using Machine Learning"
)

name = st.text_input(
    "Enter Name",
    placeholder="Naaz"
)

if st.button("Predict Gender"):

    if name:

        with st.spinner(
            "Predicting..."
        ):

            response = requests.post(
                f"{API_URL}/predict-gender",
                json={
                    "name": name
                }
            )

            result = response.json()

            if "predicted_gender" in result:

                gender = result[
                    "predicted_gender"
                ]

                st.success(
                    f"Predicted Gender: {gender.title()}"
                )

                st.metric(
                    "Prediction",
                    gender.title()
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
st.markdown(
    "Discover hotels based on city and budget"
)

col1, col2 = st.columns(2)

with col1:

    city = st.text_input(
        "City",
        placeholder="Salvador (BH)"
    )

with col2:

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

    with st.spinner(
        "Searching hotels..."
    ):

        response = requests.post(
            f"{API_URL}/recommend-hotels",
            json=payload
        )

        hotels = response.json()

        if len(hotels) > 0:

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

            st.subheader(
                "Hotel Cards"
            )

            for hotel in hotels:

                with st.container():

                    st.markdown("---")

                    c1, c2, c3 = st.columns(
                        [3, 2, 1]
                    )

                    with c1:
                        st.write(
                            f"🏨 {hotel['name']}"
                        )

                    with c2:
                        st.write(
                            f"📍 {hotel['place']}"
                        )

                    with c3:
                        st.write(
                            f"${hotel['price']}"
                        )

                    st.write(
                        f"🛏️ Stay Duration: {hotel['days']} days"
                    )

        else:

            st.warning(
                "No hotels found for selected filters."
            )
```

st.sidebar.markdown("---")
st.sidebar.info(
"Built with Flask API + Streamlit + Machine Learning"
)

