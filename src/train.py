import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from .data_loader import load_data
from .preprocessing import prepare_data, create_preprocessor, split_data
from .config import MODEL_PATH


def train_model():
    print("Loading dataset...")
    df = load_data()

    print("\nPreparing data...")
    X, y = prepare_data(df)

    print("\nFeatures used by the model:")
    print(X.columns.tolist())

    print("\nDataset shape:", X.shape)
    print("Target shape:", y.shape)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    preprocessor = create_preprocessor()

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("neural_network", model)
    ])

    print("\nTraining Neural Network...")
    pipeline.fit(X_train, y_train)

    print("\nTraining completed successfully.")

    joblib.dump(pipeline, MODEL_PATH)

    print("\nModel saved at:")
    print(MODEL_PATH)


if __name__ == "__main__":
    train_model()