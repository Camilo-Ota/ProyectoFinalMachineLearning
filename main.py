
import pandas as pd

from src.data.load_data import load_raw_data
from src.features.features_engineering import feature_engineering

from src.models.train import (
    train_logistic,
    train_random_forest,
    train_xgboost,
    train_xgboost_ultra,
    split_data
)

from src.models.evaluate import evaluar_overfitting


RANDOM_STATE = 42


# TRAINING WRAPPER
def train_all_models(X_train, X_test, y_train, y_test):

    models = {}

    print("\nEntrenando Logistic")
    model, metrics, overfit = train_logistic(X_train, X_test, y_train, y_test)
    models["Logistic"] = {
        "model": model,
        "metrics": metrics,
        "overfitting": overfit
    }

    print("\nEntrenando Random Forest")
    model, metrics, overfit = train_random_forest(X_train, X_test, y_train, y_test)
    models["RandomForest"] = {
        "model": model,
        "metrics": metrics,
        "overfitting": overfit
    }

    print("\nEntrenando XGBoost")
    model, metrics, overfit = train_xgboost(X_train, X_test, y_train, y_test)
    models["XGBoost"] = {
        "model": model,
        "metrics": metrics,
        "overfitting": overfit
    }

    print("\n Entrenando XGBoost Ultra ")
    model, metrics, overfit = train_xgboost_ultra(X_train, X_test, y_train, y_test)
    models["XGBoost_Ultra"] = {
        "model": model,
        "metrics": metrics,
        "overfitting": overfit
    }

    return models



# TABLA DE RESULTADOS
def build_results_table(models_dict):

    rows = []

    for name, info in models_dict.items():

        row = {
            "model": name,
            **info["metrics"],
            **info["overfitting"]
        }

        rows.append(row)

    df_results = pd.DataFrame(rows)
    return df_results.sort_values("recall", ascending=False)



# OVERFITTING 
def evaluate_all(models_dict, X_train, X_test, y_train, y_test):

    print("\n EVALUACIÓN DE OVERFITTING")

    for name, info in models_dict.items():
        print(f"\nModelo: {name}")
        evaluar_overfitting(
            info["model"],
            X_train,
            X_test,
            y_train,
            y_test,
            name
        )


# MAIN PIPELINE
def main():

    print("\nINICIANDO PIPELINE BNPL")

    df = load_raw_data()
    print(f"Dataset cargado: {df.shape}")

    # FEATURE ENGINEERING
    df_model = feature_engineering(df)

    # SPLIT
    X = df_model.drop("default_flag", axis=1)
    y = df_model["default_flag"]

    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    # TRAIN
    models = train_all_models(X_train, X_test, y_train, y_test)


    # RESULTS TABLE
    df_results = build_results_table(models)

    print("\nRESULTADOS")
    print(df_results)

    # SAVE RESULTS
    df_results.to_csv("reports/model_results.csv", index=False)
    print("\nResultados guardados en reports/model_results.csv")

    # OVERFITTING 
    evaluate_all(models, X_train, X_test, y_train, y_test)

    print("\nPIPELINE COMPLETADO")

if __name__ == "__main__":
    main()