from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# MÉTRICAS PRINCIPALES
def calcular_metricas(y_true, y_pred, y_prob):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }


# FORMATO BONITO PARA IMPRIMIR LAS MÉTRICAS

def imprimir_metricas(metricas: dict, nombre_modelo="Modelo"):

    print(f"\n{nombre_modelo}")

    for k, v in metricas.items():
        print(f"{k}: {round(v, 5)}")