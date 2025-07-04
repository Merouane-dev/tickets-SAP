# File: api/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# Load model & encoder
model = joblib.load("models/linear_regression.joblib")
encoder = joblib.load("models/encoder.joblib")

# Source de données (à adapter selon projet réel : DB, API...)
TICKETS_CSV = "data/tickets_cleaned.csv"
tickets_df = pd.read_csv(TICKETS_CSV)  # contient toutes les colonnes utiles

@app.route("/predict", methods=["GET"])
def predict():
    try:
        ticket_id = request.args.get("id", type=int)

        if ticket_id is None:
            return jsonify({"error": "Paramètre 'id' manquant"}), 400

        # Cherche le ticket
        ticket = tickets_df[tickets_df["ID_TICKET"] == ticket_id]
        if ticket.empty:
            return jsonify({"error": f"Ticket {ticket_id} introuvable"}), 404

        # Encodage & prédiction
        features = encoder.transform(ticket)
        prediction = model.predict(features)[0]

        return jsonify({
            "ID_TICKET": int(ticket_id),
            "PredictedDuration": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
