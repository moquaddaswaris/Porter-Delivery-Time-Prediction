import pandas as pd
from .config import DATA_PATH


def load_data():
    df = pd.read_csv(DATA_PATH)

    print("Dataset Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    return df