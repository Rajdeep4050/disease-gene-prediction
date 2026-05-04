import pandas as pd
from datetime import datetime
import os

# ======================
# Paths
# ======================
EXPERIMENT_RESULTS_PATH = "models/experiment_results.csv"
HUB_BIAS_RESULTS_PATH = "models/hub_bias_results.csv"


# ======================
# 🧪 MODEL EXPERIMENT TRACKING
# ======================
def log_experiment(name, features, precision, recall, f1, notes=""):
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": name,
        "features": ", ".join(features),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "notes": notes
    }

    if os.path.exists(EXPERIMENT_RESULTS_PATH):
        df = pd.read_csv(EXPERIMENT_RESULTS_PATH)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(EXPERIMENT_RESULTS_PATH, index=False)

    print(f"✅ Logged experiment: {name}")


# ======================
# 📊 HUB BIAS TRACKING
# ======================
def log_hub_bias(bin_name, precision, recall, f1, samples, model_name=""):
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "degree_bin": bin_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "samples": samples
    }

    if os.path.exists(HUB_BIAS_RESULTS_PATH):
        df = pd.read_csv(HUB_BIAS_RESULTS_PATH)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(HUB_BIAS_RESULTS_PATH, index=False)

    print(f"📊 Logged hub bias: {bin_name}")