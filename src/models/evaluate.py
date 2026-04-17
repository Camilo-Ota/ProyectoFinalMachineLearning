from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)


# MÉTRICAS PRINCIPALES

def evaluar_modelo(y_test, y_pred, y_prob, nombre):

    print(f"\n--- {nombre} ---")

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("Accuracy:", round(acc, 4))
    print("Recall:", round(recall, 4))
    print("Precision:", round(precision, 4))
    print("F1-score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    return {
        "modelo": nombre,
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "roc_auc": auc
    }



# OVERFITTING CHECK
def evaluar_overfitting(model, X_train, X_test, y_train, y_test, nombre):

    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_prob)
    test_auc = roc_auc_score(y_test, y_test_prob)

    print(f"\n--- Overfitting: {nombre} ---")
    print("Train ROC-AUC:", round(train_auc, 4))
    print("Test ROC-AUC:", round(test_auc, 4))
    print("Diferencia:", round(train_auc - test_auc, 4))

    return {
        "modelo": nombre,
        "train_auc": train_auc,
        "test_auc": test_auc,
        "overfitting_gap": train_auc - test_auc
    }