import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

from src.model.feature_importance import get_feature_importance
from src.model.save_model import save_model
from src.model.experiment_tracker import log_experiment, log_hub_bias
from src.model.evaluation import evaluate_model


def train_models(df: pd.DataFrame):

    print("\n===== TRAINING EXPERIMENTS =====")

    # ======================
    # Feature Engineering
    # ======================
    df["log_degree_gene1"] = np.log1p(df["degree_gene1"])
    df["log_degree_gene2"] = np.log1p(df["degree_gene2"])

    df["degree_diff"] = abs(df["degree_gene1"] - df["degree_gene2"])
    df["degree_sum"] = df["degree_gene1"] + df["degree_gene2"]
    df["preferential_attachment"] = df["degree_gene1"] * df["degree_gene2"]

    # ======================
    # Experiments
    # ======================
    experiments = {
        "baseline": [
            "log_degree_gene1",
            "log_degree_gene2",
            "common_neighbors",
            "jaccard_similarity"
        ],
        "enhanced_features": [
            "log_degree_gene1",
            "log_degree_gene2",
            "common_neighbors",
            "jaccard_similarity",
            "degree_diff",
            "degree_sum",
            "preferential_attachment"
        ],
        "no_degree": [
            "common_neighbors",
            "jaccard_similarity"
        ]
    }

    all_results = []

    for exp_name, features in experiments.items():

        print(f"\n🚀 Running Experiment: {exp_name}")

        X = df[features]
        y = df["edge_label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ======================
        # Logistic Regression
        # ======================
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)

        eval_lr = evaluate_model(lr, X_test, y_test, f"{exp_name}_LR")

        log_experiment(
            name=f"{exp_name}_LR",
            features=features,
            precision=eval_lr["precision"],
            recall=eval_lr["recall"],
            f1=eval_lr["f1"],
            notes="Logistic Regression"
        )

        all_results.append({
            "model": "LR",
            "experiment": exp_name,
            "features": features,
            "f1": eval_lr["f1"],
            "model_obj": lr
        })

        # ======================
        # Random Forest
        # ======================
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42
        )

        rf.fit(X_train, y_train)

        eval_rf = evaluate_model(rf, X_test, y_test, f"{exp_name}_RF")

        log_experiment(
            name=f"{exp_name}_RF",
            features=features,
            precision=eval_rf["precision"],
            recall=eval_rf["recall"],
            f1=eval_rf["f1"],
            notes="Random Forest"
        )

        all_results.append({
            "model": "RF",
            "experiment": exp_name,
            "features": features,
            "f1": eval_rf["f1"],
            "model_obj": rf
        })

        # ======================
        # XGBoost
        # ======================
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(len(y_train) / sum(y_train)),
            random_state=42,
            eval_metric="logloss"
        )

        xgb.fit(X_train, y_train)

        eval_xgb = evaluate_model(xgb, X_test, y_test, f"{exp_name}_XGB")

        log_experiment(
            name=f"{exp_name}_XGB",
            features=features,
            precision=eval_xgb["precision"],
            recall=eval_xgb["recall"],
            f1=eval_xgb["f1"],
            notes="XGBoost"
        )

        all_results.append({
            "model": "XGB",
            "experiment": exp_name,
            "features": features,
            "f1": eval_xgb["f1"],
            "model_obj": xgb
        })

    # ======================
    # BEST MODEL SELECTION (FIXED)
    # ======================
    best_entry = max(all_results, key=lambda x: x["f1"])

    best_model = best_entry["model_obj"]
    best_features = best_entry["features"]
    best_model_name = f"{best_entry['experiment']}_{best_entry['model']}"
    
    # Rebuild the SAME split for the best experiment (deterministic because random_state=42)
    X = df[best_features]
    y = df["edge_label"]

    _, X_test_best, _, y_test_best = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n🏆 BEST MODEL")
    print(f"Model: {best_model_name}")
    print(f"F1: {best_entry['f1']:.3f}")

    save_model(best_model)

    return best_model, best_features, best_model_name, X_test_best, y_test_best

def hub_bias_analysis(best_model, df, best_features, best_model_name, best_threshold):

    print("\n===== HUB BIAS ANALYSIS =====")

    X = df[best_features]
    y = df["edge_label"]

    preds = predict_with_threshold(best_model, X, best_threshold)

    df_copy = df.copy()
    df_copy["prediction"] = preds

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

        from sklearn.metrics import classification_report
        report = classification_report(
            subset["edge_label"],
            subset["prediction"],
            output_dict=True
        )

        log_hub_bias(
            bin_name=bin_name,
            precision=report["1"]["precision"],
            recall=report["1"]["recall"],
            f1=report["1"]["f1-score"],
            samples=len(subset),
            model_name=best_model_name
        )


def find_best_threshold(model, X_test, y_test):

    probs = model.predict_proba(X_test)[:, 1]

    best_f1 = 0
    best_threshold = 0.5

    print("\n===== THRESHOLD TUNING =====")

    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:

        preds = (probs >= t).astype(int)

        report = classification_report(y_test, preds, output_dict=True)

        f1 = report["1"]["f1-score"]

        print(f"Threshold: {t} → F1: {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"\n🏆 Best Threshold: {best_threshold}")
    print(f"Best F1: {best_f1:.3f}")

    return best_threshold, best_f1

def predict_with_threshold(model, X, threshold):
    probs = model.predict_proba(X)[:, 1]
    return (probs >= threshold).astype(int)

if __name__ == "__main__":

    print("===== LOADING DATA =====")

    df = pd.read_csv("data/processed/final_dataset.csv")

    best_model, best_features, best_model_name, X_test, y_test = train_models(df)
    
    best_threshold, best_f1 = find_best_threshold(best_model, X_test, y_test)

    hub_bias_analysis(best_model, df, best_features, best_model_name, best_threshold)

    # ======================
    # 🚀 XGBOOST TUNING (FINAL STEP)
    # ======================

    print("\n===== FINAL XGBOOST TUNING =====")

    # Recreate TRAIN split (important!)
    X = df[best_features]
    y = df["edge_label"]

    X_train, X_test_xgb, y_train, y_test_xgb = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 5, 10]
    }

    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    tuned_xgb = search.best_estimator_

    print("\n🏆 BEST XGB PARAMS:")
    print(search.best_params_)

    # Evaluate tuned XGB
    eval_xgb = evaluate_model(tuned_xgb, X_test_xgb, y_test_xgb, "TUNED_XGB")

    # Threshold tuning for XGB
    best_threshold_xgb, best_f1_xgb = find_best_threshold(
        tuned_xgb, X_test_xgb, y_test_xgb
    )

    final_threshold = best_threshold_xgb

    joblib.dump({
        "model": tuned_xgb,
        "threshold": best_threshold_xgb
    }, "models/xgb_with_threshold.pkl")
    
    print("\n===== FINAL COMPARISON =====")
    print(f"RF (threshold) → F1: {best_f1:.3f}")
    print(f"XGB (threshold) → F1: {best_f1_xgb:.3f}")