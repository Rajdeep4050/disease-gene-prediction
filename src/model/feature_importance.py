import pandas as pd


def get_feature_importance(model, feature_names):
    """
    Extract and display feature importance from trained model.

    Args:
        model: trained RandomForest model
        feature_names: list of feature column names
    """

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\n===== FEATURE IMPORTANCE (Random Forest) =====")
    print(importance_df)

    return importance_df