import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from .data_loader import load_data
from .preprocessing import prepare_data, split_data
from .config import MODEL_PATH


def evaluate_model():
    print("Loading dataset...")
    df = load_data()

    print("\nPreparing test data...")
    X, y = prepare_data(df)

    _, X_test, _, y_test = split_data(X, y)

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n========== MODEL EVALUATION ==========")
    print(f"MAE  : {mae:.2f} minutes")
    print(f"RMSE : {rmse:.2f} minutes")
    print(f"R²   : {r2:.4f}")
    print("=======================================")


if __name__ == "__main__":
    evaluate_model()