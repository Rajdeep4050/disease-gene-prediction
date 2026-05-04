import pandas as pd
from datetime import datetime
import os

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score
)

EVAL_PATH = "models/evaluation_results.csv"


def evaluate_model(model, X_test, y_test, model_name="model"):
    """
    Evaluate model with classification + ROC-AUC + PR-AUC
    """

    # Probabilities
    probs = model.predict_proba(X_test)[:, 1]

    # Default predictions (threshold = 0.5)
    preds = (probs >= 0.5).astype(int)

    report = classification_report(y_test, preds, output_dict=True)

    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]

    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)

    print("\n===== MODEL EVALUATION =====")
    print(f"Model: {model_name}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}")

    # Save results
    log_evaluation(
        model_name,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "probs": probs
    }


def log_evaluation(model_name, precision, recall, f1, roc_auc, pr_auc):
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

    if os.path.exists(EVAL_PATH):
        df = pd.read_csv(EVAL_PATH)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(EVAL_PATH, index=False)

    print("📊 Evaluation logged")