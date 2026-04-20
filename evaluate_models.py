import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
)


def find_best_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    best_idx = int(np.argmax(f1_scores))
    if best_idx >= len(thresholds):
        best_threshold = 0.5
    else:
        best_threshold = float(thresholds[best_idx])

    return best_threshold


def compute_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def evaluate_all_models(models, X_val, y_val, X_test, y_test):
    validation_rows = []
    test_rows = []
    best_model_name = None
    best_model = None
    best_params = None
    best_threshold = None
    best_val_pr_auc = -1

    for name, payload in models.items():
        model = payload["model"]
        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = find_best_threshold(y_val, val_proba)

        val_metrics = compute_metrics(y_val, val_proba, threshold)
        validation_rows.append({"model": name, **val_metrics})

        test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, test_proba, threshold)
        test_rows.append({"model": name, **test_metrics})

        if val_metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics["pr_auc"]
            best_model_name = name
            best_model = model
            best_params = payload["params"]
            best_threshold = threshold

    validation_df = pd.DataFrame(validation_rows).sort_values("pr_auc", ascending=False)
    test_df = pd.DataFrame(test_rows).sort_values("pr_auc", ascending=False)

    return {
        "validation_df": validation_df,
        "test_df": test_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_params": best_params,
        "best_threshold": best_threshold,
    }
