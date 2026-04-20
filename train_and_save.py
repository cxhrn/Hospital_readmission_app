from data_preprocessing import prepare_data
from train_models import train_all_models
from evaluate_models import evaluate_all_models
from config import MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE
from utils import save_pickle, save_json, set_seed

def main():
    set_seed(RANDOM_STATE)

    print("=" * 60)
    print("Preparing data")
    print("=" * 60)
    data = prepare_data()

    print("\n" + "=" * 60)
    print("Training candidate models")
    print("=" * 60)
    models, tuning_df = train_all_models(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    )

    tuning_df.to_csv(OUTPUTS_DIR / "tuning_results.csv", index=False)

    print("\n" + "=" * 60)
    print("Evaluating models")
    print("=" * 60)
    results = evaluate_all_models(
        models,
        data["X_val"], data["y_val"],
        data["X_test"], data["y_test"]
    )

    results["validation_df"].to_csv(OUTPUTS_DIR / "validation_metrics.csv", index=False)
    results["test_df"].to_csv(OUTPUTS_DIR / "test_metrics.csv", index=False)

    best_model = results["best_model"]
    best_model_name = results["best_model_name"]
    best_threshold = results["best_threshold"]

    best_test_row = results["test_df"][results["test_df"]["model"] == best_model_name].iloc[0]

    save_pickle(best_model, MODELS_DIR / "best_model.pkl")
    save_pickle(data["preprocessor"], MODELS_DIR / "preprocessor.pkl")

    metadata = {
        "best_model_name": best_model_name,
        "best_threshold": float(best_threshold),
        "pr_auc": float(best_test_row["pr_auc"]),
        "roc_auc": float(best_test_row["roc_auc"]),
        "precision": float(best_test_row["precision"]),
        "recall": float(best_test_row["recall"]),
        "f1": float(best_test_row["f1"]),
        "balanced_accuracy": float(best_test_row["balanced_accuracy"]),
        "best_params": results["best_params"],
    }
    save_json(metadata, MODELS_DIR / "best_model_metadata.json")

    print("\nSaved files in models/:")
    print("- best_model.pkl")
    print("- preprocessor.pkl")
    print("- best_model_metadata.json")
    print(f"\nBest model: {best_model_name}")
    print(f"Best threshold: {best_threshold:.4f}")

if __name__ == "__main__":
    main()
