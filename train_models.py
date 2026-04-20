import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from config import RANDOM_STATE


def scale_pos_weight_from_y(y):
    neg = np.sum(y == 0)
    pos = np.sum(y == 1)
    return neg / pos if pos > 0 else 1.0


def train_logistic_regression_candidates(X_train, y_train, X_val, y_val):
    candidates = [0.01, 0.1, 1.0, 5.0]
    records = []
    best_model = None
    best_score = -1
    best_params = None

    for c in candidates:
        model = LogisticRegression(
            C=c,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=500,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, val_proba)

        records.append({"model": "Logistic Regression", "C": c, "validation_pr_auc": pr_auc})

        if pr_auc > best_score:
            best_score = pr_auc
            best_model = model
            best_params = {"C": c}

    return best_model, best_params, pd.DataFrame(records)


def train_random_forest_candidates(X_train, y_train, X_val, y_val):
    grid = [
        {"n_estimators": 150, "max_depth": 12},
        {"n_estimators": 200, "max_depth": 15},
        {"n_estimators": 250, "max_depth": None},
    ]

    records = []
    best_model = None
    best_score = -1
    best_params = None

    for params in grid:
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, val_proba)

        records.append({
            "model": "Random Forest",
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "validation_pr_auc": pr_auc
        })

        if pr_auc > best_score:
            best_score = pr_auc
            best_model = model
            best_params = params

    return best_model, best_params, pd.DataFrame(records)


def train_xgboost_candidates(X_train, y_train, X_val, y_val):
    spw = scale_pos_weight_from_y(y_train)
    grid = [
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05},
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03},
        {"n_estimators": 700, "max_depth": 5, "learning_rate": 0.03},
    ]

    records = []
    best_model = None
    best_score = -1
    best_params = None

    for params in grid:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=30,
            **params,
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, val_proba)

        records.append({
            "model": "XGBoost",
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"],
            "validation_pr_auc": pr_auc,
            "best_iteration": getattr(model, "best_iteration", None),
        })

        if pr_auc > best_score:
            best_score = pr_auc
            best_model = model
            best_params = {
                **params,
                "scale_pos_weight": spw,
                "best_iteration": getattr(model, "best_iteration", None),
            }

    return best_model, best_params, pd.DataFrame(records)


def train_all_models(X_train, y_train, X_val, y_val):
    lr_model, lr_params, lr_df = train_logistic_regression_candidates(X_train, y_train, X_val, y_val)
    rf_model, rf_params, rf_df = train_random_forest_candidates(X_train, y_train, X_val, y_val)
    xgb_model, xgb_params, xgb_df = train_xgboost_candidates(X_train, y_train, X_val, y_val)

    tuning_df = pd.concat([lr_df, rf_df, xgb_df], ignore_index=True)

    models = {
        "Logistic Regression": {"model": lr_model, "params": lr_params},
        "Random Forest": {"model": rf_model, "params": rf_params},
        "XGBoost": {"model": xgb_model, "params": xgb_params},
    }

    return models, tuning_df
