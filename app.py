
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# ==================================================
# LOAD MODELS
# ==================================================

flight_model = joblib.load("/content/drive/MyDrive/Voyage Analytics: Integrating MLOps in Travel/models/flight_price_model.pkl")

flight_encoders = joblib.load(
    "/content/drive/MyDrive/Voyage Analytics: Integrating MLOps in Travel/models/flight_encoders.pkl"
)

gender_model_data = joblib.load(
    "/content/drive/MyDrive/Voyage Analytics: Integrating MLOps in Travel/models/name_gender_classifier.pkl"
)

gender_classifier = gender_model_data["classifier"]

gender_encoder = gender_model_data["label_encoder"]

embedding_model = SentenceTransformer(
    gender_model_data["model_name"]
)



user_similarity_df = joblib.load(
    "/content/drive/MyDrive/Voyage Analytics: Integrating MLOps in Travel/models/user_similarity.pkl"
)

user_hotel_matrix = joblib.load(
    "/content/drive/MyDrive/Voyage Analytics: Integrating MLOps in Travel/models/user_hotel_matrix.pkl"
)

hotel_df = pd.read_csv("/content/drive/MyDrive/Voyage Analytics: Integrating MLOps in Travel/data/hotels.csv")

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

    flight_type_num = flight_encoders["flightType"].transform(
        [flight_type]
    )[0]

    agency_num = flight_encoders["agency"].transform(
        [agency]
    )[0]

    input_data = pd.DataFrame([{
        "from_num": from_num,
        "to_num": to_num,
        "flightType_num": flight_type_num,
        "agency_num": agency_num,
        "time": time,
        "distance": distance
    }])

    prediction = flight_model.predict(
        input_data
    )[0]

    return jsonify({
        "predicted_price": round(float(prediction), 2)
    })

# ==================================================
# GENDER PREDICTION
# ==================================================

@app.route("/predict-gender", methods=["POST"])
def predict_gender():

    try:

        data = request.json

        name = data["name"]

        first_name = name.split()[0]

        embedding = embedding_model.encode(
            [first_name]
        )

        prediction = gender_classifier.predict(
            embedding
        )[0]

        gender = gender_encoder.inverse_transform(
            [prediction]
        )[0]

        return jsonify({
            "name": name,
            "predicted_gender": str(gender)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ==================================================
# HOTEL RECOMMENDATION
# ==================================================

def recommend_hotels(user_id, n_recommendations=5):

    similar_users = (
        user_similarity_df[user_id]
        .sort_values(ascending=False)
        .iloc[1:6]
        .index
    )

    user_hotels = set(
        hotel_df[
            hotel_df["userCode"] == user_id
        ]["hotel_name"]
    )

    recommendations = []

    for sim_user in similar_users:

        hotels = hotel_df[
            hotel_df["userCode"] == sim_user
        ]["hotel_name"]

        for hotel in hotels:

            if hotel not in user_hotels:
                recommendations.append(hotel)

    return list(set(recommendations))[:n_recommendations]

@app.route("/recommend-hotels", methods=["POST"])
def recommend():

    data = request.json

    user_id = int(data["user_id"])

    city = data.get("city")

    min_price = float(
        data.get("min_price", 0)
    )

    max_price = float(
        data.get("max_price", 10000)
    )

    recommended = recommend_hotels(user_id)

    results = hotel_df[
        hotel_df["hotel_name"].isin(recommended)
    ]

    if city:
        results = results[
            results["place"] == city
        ]

    results = results[
        (results["hotel_price"] >= min_price)
        &
        (results["hotel_price"] <= max_price)
    ]

    output = results[
        [
            "hotel_name",
            "place",
            "hotel_price",
            "days"
        ]
    ].drop_duplicates()

    return jsonify(
        output.to_dict(orient="records")
    )

# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
