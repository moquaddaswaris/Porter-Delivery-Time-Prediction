import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = [
    "market_id",
    "order_protocol",
    "total_items",
    "subtotal"
]

CATEGORICAL_FEATURES = ["store_primary_category"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "delivery_time"


def prepare_data(df):
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["actual_delivery_time"] = pd.to_datetime(
        df["actual_delivery_time"],
        errors="coerce"
    )

    df[TARGET] = (
        df["actual_delivery_time"] - df["created_at"]
    ).dt.total_seconds() / 60

    df = df.dropna(subset=FEATURES + [TARGET])

    df = df[
        (df[TARGET] > 0) &
        (df[TARGET] < 300)
    ]

    X = df[FEATURES]
    y = df[TARGET]

    return X, y


def create_preprocessor():
    return ColumnTransformer([
        (
            "numeric",
            StandardScaler(),
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            CATEGORICAL_FEATURES
        )
    ])


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )