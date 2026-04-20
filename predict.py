import pandas as pd

from config import APP_COLUMNS, MODELS_DIR
from utils import load_pickle, load_json


def load_artifacts():
    model = load_pickle(MODELS_DIR / "best_model.pkl")
    preprocessor = load_pickle(MODELS_DIR / "preprocessor.pkl")
    metadata = load_json(MODELS_DIR / "best_model_metadata.json")
    return model, preprocessor, metadata


def build_input_row(user_inputs):
    row = {col: user_inputs.get(col, None) for col in APP_COLUMNS}
    return pd.DataFrame([row])


def make_prediction(user_inputs):
    model, preprocessor, metadata = load_artifacts()
    input_df = build_input_row(user_inputs)
    X_processed = preprocessor.transform(input_df)

    proba_pos = float(model.predict_proba(X_processed)[0, 1])
    proba_neg = 1.0 - proba_pos
    threshold = float(metadata["best_threshold"])
    predicted_class = int(proba_pos >= threshold)

    return {
        "predicted_probability": proba_pos,
        "negative_probability": proba_neg,
        "model_confidence": max(proba_pos, proba_neg),
        "threshold": threshold,
        "prediction": predicted_class,
        "model_name": metadata["best_model_name"],
        "metadata": metadata,
    }
