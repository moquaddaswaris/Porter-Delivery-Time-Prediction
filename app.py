from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        market_id = int(request.form["market_id"])
        order_protocol = int(request.form["order_protocol"])
        total_items = int(request.form["total_items"])
        subtotal = float(request.form["subtotal"])
        store_primary_category = request.form["store_primary_category"]

        input_data = pd.DataFrame([{
            "market_id": market_id,
            "order_protocol": order_protocol,
            "total_items": total_items,
            "subtotal": subtotal,
            "store_primary_category": store_primary_category
        }])

        prediction = model.predict(input_data)[0]
        prediction = max(0, round(float(prediction), 2))

        return render_template(
            "index.html",
            prediction=prediction
        )

    except Exception:
        return render_template(
            "index.html",
            error="Please enter valid values."
        )


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )