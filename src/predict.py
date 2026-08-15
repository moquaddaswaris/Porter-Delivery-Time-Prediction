import joblib
from .config import MODEL_PATH

def predict_delivery_time(data):
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([data])
    return round(float(prediction[0]), 2)


if __name__ == "__main__":

    sample = {
        "market_id": 1,
        "order_protocol": 3,
        "total_items": 3,
        "subtotal": 450,
        "num_distinct_items": 3,
        "min_item_price": 80,
        "max_item_price": 200,
        "total_onshift_partners": 10,
        "total_busy_partners": 6,
        "total_outstanding_orders": 8,
        "store_primary_category": "American"
    }

    print(
        "Predicted delivery time:",
        predict_delivery_time(sample),
        "minutes"
    )