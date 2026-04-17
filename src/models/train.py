import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src.models.evaluate import evaluar_modelo, evaluar_overfitting



# SPLIT DATA
def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# LOGISTIC REGRESSION
def train_logistic(X_train, X_test, y_train, y_test):
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight='balanced',
            max_iter=2000
        ))
    ])

    # Entrenamiento
    pipeline.fit(X_train, y_train)
    

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metricas=evaluar_modelo(y_test, y_pred, y_prob, "Logistic Regression")
    overfit=evaluar_overfitting(pipeline, X_train, X_test, y_train, y_test, "Logistic Regression")


    return pipeline, metricas,overfit



# RANDOM FOREST
def train_random_forest(X_train, X_test, y_train, y_test):

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metricas=evaluar_modelo(y_test, y_pred, y_prob, "Random Forest")
    overfit=evaluar_overfitting(model, X_train, X_test, y_train, y_test, "Random Forest")


    return model, metricas,overfit


# XGBOOST
def train_xgboost(X_train, X_test, y_train, y_test):

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos

    model = xgb.XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=scale,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metricas=evaluar_modelo(y_test, y_pred, y_prob, "XGBoost")
    overfit=evaluar_overfitting(model, X_train, X_test, y_train, y_test, "XGBoost")


    return model, metricas,overfit


# XGBOOST Con mejores hiperparámetros

def train_xgboost_ultra(X_train, X_test, y_train, y_test):

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos

    model = xgb.XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=scale,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metricas=evaluar_modelo(y_test, y_pred, y_prob, "XGBoost-Ultra")
    overfit=evaluar_overfitting(model, X_train, X_test, y_train, y_test, "XGBoost-Ultra")


    return model, metricas,overfit



# PIPELINE PRINCIPAL
def train_all_models(feature_sets: dict, y):

    results = {}

    for name, X in feature_sets.items():

        print(f"\n DATASET: {name} ")

        X_train, X_test, y_train, y_test = split_data(X, y)

        models = {}

        # Logistic
        model, metrics, overfit = train_logistic(X_train, X_test, y_train, y_test)
        models["logistic"] = {
            "model": model,
            "metrics": metrics,
            "overfitting": overfit
        }

        # Random Forest
        model, metrics, overfit = train_random_forest(X_train, X_test, y_train, y_test)
        models["rf"] = {
            "model": model,
            "metrics": metrics,
            "overfitting": overfit
        }

        # XGBoost
        model, metrics, overfit = train_xgboost(X_train, X_test, y_train, y_test)
        models["xgb"] = {
            "model": model,
            "metrics": metrics,
            "overfitting": overfit
        }

        # XGBoost Ultra
        model, metrics, overfit = train_xgboost_ultra(X_train, X_test, y_train, y_test)
        models["xgb_ultra"] = {
            "model": model,
            "metrics": metrics,
            "overfitting": overfit
        }

        results[name] = models

    return results