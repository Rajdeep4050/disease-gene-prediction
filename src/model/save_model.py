import joblib
import os


def save_model(model, path="models/random_forest.pkl"):
    """
    Save trained model to disk.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

    print(f"\nModel saved at: {path}")