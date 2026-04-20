import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    DATA_PATH,
    DEATH_CODES,
    DROP_COLS,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
)


def load_and_clean_data(filepath=DATA_PATH):
    print("Loading dataset...")
    df = pd.read_csv(filepath, on_bad_lines="skip", na_values="?")
    print(f"Original shape: {df.shape}")

    df = df[~df["discharge_disposition_id"].isin(DEATH_CODES)]
    df = df[df["gender"] != "Unknown/Invalid"]

    df[TARGET_COL] = (df["readmitted"] == "<30").astype(int)

    existing_drop_cols = [col for col in DROP_COLS if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)

    print(f"Cleaned shape: {df.shape}")
    print(f"Readmission rate: {df[TARGET_COL].mean():.4f}")
    return df


def build_preprocessor(X):
    numerical_features = [col for col in NUMERICAL_FEATURES if col in X.columns]
    categorical_features = [col for col in X.columns if col not in numerical_features]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numerical_features),
        ("cat", cat_pipe, categorical_features)
    ])

    return preprocessor, numerical_features, categorical_features


def prepare_data():
    df = load_and_clean_data()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 60/20/20 stratified split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=RANDOM_STATE
    )

    preprocessor, num_features, cat_features = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = num_features + list(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(cat_features)
    )

    print("\nSplit summary:")
    print(f"Train: {X_train.shape[0]} rows | Rate: {y_train.mean():.4f}")
    print(f"Val:   {X_val.shape[0]} rows | Rate: {y_val.mean():.4f}")
    print(f"Test:  {X_test.shape[0]} rows | Rate: {y_test.mean():.4f}")

    return {
        "X_train_raw": X_train,
        "X_val_raw": X_val,
        "X_test_raw": X_test,
        "X_train": X_train_processed,
        "X_val": X_val_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }
