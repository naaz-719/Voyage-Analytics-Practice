from flask import Flask, request, jsonify
import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ==================================================
# LOAD MODELS
# ==================================================

flight_model = joblib.load(
    "models/flight_price_model.pkl"
)

flight_encoders = joblib.load(
    "models/flight_encoders.pkl"
)

gender_model = joblib.load(
    "models/gender_model.pkl"
)

gender_tfidf = joblib.load(
    "models/gender_tfidf.pkl"
)

gender_label_encoder = joblib.load(
    "models/gender_label_encoder.pkl"
)

user_hotel_matrix = joblib.load(
    "models/user_hotel_matrix.pkl"
)

hotel_df = pd.read_csv(
    "data/hotels.csv"
)

# ==================================================
# HOTEL SIMILARITY MATRIX
# ==================================================

user_similarity_df = pd.DataFrame(
    cosine_similarity(user_hotel_matrix),
    index=user_hotel_matrix.index,
    columns=user_hotel_matrix.index
)

# ==================================================
# HOME
# ==================================================

@app.route("/")
def home():

    return jsonify({
        "project": "Voyage Analytics",
        "status": "running",
        "available_endpoints": [
            "/predict-flight",
            "/predict-gender",
            "/recommend-hotels"
        ]
    })

# ==================================================
# FLIGHT PRICE PREDICTION
# ==================================================

@app.route("/predict-flight", methods=["POST"])
def predict_flight():

    try:

        data = request.json

        from_city = data["from"]
        to_city = data["to"]
        flight_type = data["flightType"]
        agency = data["agency"]
        time = float(data["time"])
        distance = float(data["distance"])

        from_num = flight_encoders["from"].transform(
            [from_city]
        )[0]

        to_num = flight_encoders["to"].transform(
            [to_city]
        )[0]

        flight_type_num = flight_encoders[
            "flightType"
        ].transform(
            [flight_type]
        )[0]

        agency_num = flight_encoders[
            "agency"
        ].transform(
            [agency]
        )[0]

        input_df = pd.DataFrame([{
            "from_num": from_num,
            "to_num": to_num,
            "flightType_num": flight_type_num,
            "agency_num": agency_num,
            "time": time,
            "distance": distance
        }])

        prediction = flight_model.predict(
            input_df
        )[0]

        return jsonify({
            "predicted_price": round(
                float(prediction), 2
            )
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ==================================================
# GENDER PREDICTION
# ==================================================

@app.route("/predict-gender", methods=["POST"])
def predict_gender():

    try:

        data = request.json

        name = data["name"]

        first_name = name.split()[0]

        features = gender_tfidf.transform(
            [first_name]
        )

        prediction = gender_model.predict(
            features
        )[0]

        gender = (
            gender_label_encoder
            .inverse_transform(
                [prediction]
            )[0]
        )

        return jsonify({
            "name": name,
            "predicted_gender": str(gender)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ==================================================
# HOTEL RECOMMENDER
# ==================================================

def recommend_hotels(
    user_id,
    n_recommendations=5
):

    if user_id not in user_similarity_df.index:

        return []

    similar_users = (
        user_similarity_df[user_id]
        .sort_values(
            ascending=False
        )
        .iloc[1:6]
        .index
    )

    user_hotels = set(

        hotel_df[
            hotel_df["userCode"]
            == user_id
        ]["name"]

    )

    recommendations = []

    for sim_user in similar_users:

        hotels = hotel_df[
            hotel_df["userCode"]
            == sim_user
        ]["name"]

        for hotel in hotels:

            if hotel not in user_hotels:

                recommendations.append(
                    hotel
                )

    return list(
        set(recommendations)
    )[:n_recommendations]

# ==================================================
# HOTEL ENDPOINT
# ==================================================

@app.route(
    "/recommend-hotels",
    methods=["POST"]
)
def recommend():

    try:

        data = request.json

        user_id = int(
            data["user_id"]
        )

        city = data.get("city")

        min_price = float(
            data.get(
                "min_price",
                0
            )
        )

        max_price = float(
            data.get(
                "max_price",
                10000
            )
        )

        recommended = recommend_hotels(
            user_id
        )

        results = hotel_df[
            hotel_df[
                "name"
            ].isin(
                recommended
            )
        ]

        if city:

            results = results[
                results["place"]
                == city
            ]

        results = results[
            (
                results[
                    "price"
                ]
                >= min_price
            )
            &
            (
                results[
                    "price"
                ]
                <= max_price
            )
        ]

        output = results[
            [
                "hotel_name",
                "place",
                "price",
                "days"
            ]
        ].drop_duplicates()

        return jsonify(
            output.to_dict(
                orient="records"
            )
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
