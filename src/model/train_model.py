import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from src.model.feature_importance import get_feature_importance
from src.model.save_model import save_model

def train_models(df: pd.DataFrame):
    """
    Train baseline and advanced models.
    """

    # 🎯 Select features
    features = [
        "degree_gene1",
        "degree_gene2",
        "common_neighbors",
        "jaccard_similarity" 
    ]

    X = df[features]
    y = df["edge_label"]

    # ✅ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    # ======================
    # 🔹 Model 1: Logistic Regression
    # ======================
    print("\n===== Logistic Regression =====")

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)

    print(classification_report(y_test, y_pred_lr))

    # ======================
    # 🔹 Model 2: Random Forest
    # ======================
    print("\n===== Random Forest =====")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)

    print(classification_report(y_test, y_pred_rf))

    # Feature Importance
    
    get_feature_importance(rf, features)
    save_model(rf)

    return lr, rf, features


