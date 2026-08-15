from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

TARGET = "delivery_time"

NUMERIC_FEATURES = [
    "market_id",
    "order_protocol",
    "total_items",
    "subtotal",
    "num_distinct_items",
    "min_item_price",
    "max_item_price",
    "total_onshift_partners",
    "total_busy_partners",
    "total_outstanding_orders",
]

CATEGORICAL_FEATURES = [
    "store_primary_category"
]