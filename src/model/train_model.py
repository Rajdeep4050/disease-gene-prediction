import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.model.feature_importance import get_feature_importance
from src.model.save_model import save_model
from src.model.experiment_tracker import log_experiment, log_hub_bias

from xgboost import XGBClassifier

def train_models(df: pd.DataFrame):
    """
    Train multiple experiments (ablation study) and log results.
    """

    print("\n===== TRAINING EXPERIMENTS =====")

    # ======================
    # 🔧 Add log-degree features
    # ======================
    df["log_degree_gene1"] = np.log1p(df["degree_gene1"])
    df["log_degree_gene2"] = np.log1p(df["degree_gene2"])

    # ======================
    # 🔬 Define Experiments
    # ======================
    experiments = {
        "no_degree": [
            "common_neighbors",
            "jaccard_similarity"
        ],
        "only_degree": [
            "degree_gene1",
            "degree_gene2"
        ],
        "only_log_degree": [
            "log_degree_gene1",
            "log_degree_gene2"
        ],
        "full_with_log_degree": [
            "log_degree_gene1",
            "log_degree_gene2",
            "common_neighbors",
            "jaccard_similarity"
        ],
        "only_common_neighbors": [
            "common_neighbors"
        ],
        "only_jaccard": [
            "jaccard_similarity"
        ]
    }

    best_f1 = 0
    best_model = None
    best_features = None
    best_model_name = ""
    all_results = []

    # ======================
    # 🔁 Loop through experiments
    # ======================
    for exp_name, features in experiments.items():

        print(f"\n🚀 Running Experiment: {exp_name}")
        print(f"Features: {features}")

        X = df[features]
        y = df["edge_label"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # ======================
        # 🔹 Logistic Regression
        # ======================
        print("\n--- Logistic Regression ---")

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)

        y_pred_lr = lr.predict(X_test)
        report_lr = classification_report(y_test, y_pred_lr, output_dict=True)

        precision_lr = report_lr["1"]["precision"]
        recall_lr = report_lr["1"]["recall"]
        f1_lr = report_lr["1"]["f1-score"]

        print(f"LR -> Precision: {precision_lr:.3f}, Recall: {recall_lr:.3f}, F1: {f1_lr:.3f}")

        log_experiment(
            name=f"{exp_name}_LR",
            features=features,
            precision=precision_lr,
            recall=recall_lr,
            f1=f1_lr,
            notes="Logistic Regression"
        )

        all_results.append({
            "model": "LR",
            "experiment": exp_name,
            "features": features,
            "f1": f1_lr,
            "model_obj": lr
        })


        # ======================
        # 🔹 Random Forest
        # ======================
        print("\n--- Random Forest ---")

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42
        )

        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

        precision_rf = report_rf["1"]["precision"]
        recall_rf = report_rf["1"]["recall"]
        f1_rf = report_rf["1"]["f1-score"]

        print(f"RF -> Precision: {precision_rf:.3f}, Recall: {recall_rf:.3f}, F1: {f1_rf:.3f}")

        log_experiment(
            name=f"{exp_name}_RF",
            features=features,
            precision=precision_rf,
            recall=recall_rf,
            f1=f1_rf,
            notes="Random Forest"
        )

        all_results.append({
            "model": "RF",
            "experiment": exp_name,
            "features": features,
            "f1": f1_rf,
            "model_obj": rf
        })

        # ======================
        # 🔹 XGBoost
        # ======================
        print("\n--- XGBoost ---")

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(len(y_train) / sum(y_train)),  # handle imbalance
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False
        )

        xgb.fit(X_train, y_train)

        y_pred_xgb = xgb.predict(X_test)

        report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)

        precision_xgb = report_xgb["1"]["precision"]
        recall_xgb = report_xgb["1"]["recall"]
        f1_xgb = report_xgb["1"]["f1-score"]

        print(f"XGB -> Precision: {precision_xgb:.3f}, Recall: {recall_xgb:.3f}, F1: {f1_xgb:.3f}")

        log_experiment(
            name=f"{exp_name}_XGB",
            features=features,
            precision=precision_xgb,
            recall=recall_xgb,
            f1=f1_xgb,
            notes="XGBoost"
        )

        all_results.append({
            "model": "XGB",
            "experiment": exp_name,
            "features": features,
            "f1": f1_xgb,
            "model_obj": xgb
        })

        # Feature importance only for full model
        if exp_name == "full_with_log_degree":
            get_feature_importance(rf, features)


        # ======================
    # 🏆 Select Best Model (GLOBAL)
        # ======================
        best_entry = max(all_results, key=lambda x: x["f1"])

        best_model = best_entry["model_obj"]
        best_features = best_entry["features"]
        best_model_name = f"{best_entry['experiment']}_{best_entry['model']}"
        best_f1 = best_entry["f1"]

        print("\n🏆 BEST MODEL SELECTED")
        print(f"Model: {best_model_name}")
        print(f"F1 Score: {best_f1:.3f}")
        print(f"Features: {best_features}")

        save_model(best_model)

    return best_model, best_features, best_model_name


def hub_bias_analysis(model, df, features, model_name="RF"):

    print("\n===== HUB BIAS ANALYSIS =====")

    X = df[features]
    y = df["edge_label"]

    y_pred = model.predict(X)

    df_copy = df.copy()
    df_copy["prediction"] = y_pred

    df_copy["max_degree"] = df_copy[["degree_gene1", "degree_gene2"]].max(axis=1)

    df_copy["degree_bin"] = pd.cut(
        df_copy["max_degree"],
        bins=[0, 10, 50, 100, 1000],
        labels=["low", "medium", "high", "very_high"]
    )

    for bin_name in df_copy["degree_bin"].dropna().unique():
        subset = df_copy[df_copy["degree_bin"] == bin_name]

        if len(subset) < 100:
            continue

        print(f"\n--- Degree Bin: {bin_name} ---")
        print("Samples:", len(subset))

        report = classification_report(
            subset["edge_label"],
            subset["prediction"],
            output_dict=True
        )

        precision = report["1"]["precision"]
        recall = report["1"]["recall"]
        f1 = report["1"]["f1-score"]

        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        log_hub_bias(
            bin_name=bin_name,
            precision=precision,
            recall=recall,
            f1=f1,
            samples=len(subset),
            model_name=model_name
        )


if __name__ == "__main__":

    print("===== LOADING DATA =====")

    df = pd.read_csv("data/processed/final_dataset.csv")

    print("Dataset shape:", df.shape)

    # Train models
    best_model, best_features, best_model_name = train_models(df)

    # Hub bias analysis
    hub_bias_analysis(
        best_model,
        df,
        best_features,
        model_name=best_model_name
    )