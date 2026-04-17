import pandas as pd

def preprocess_data(df):

    df = df.copy()

    # Eliminar columnas inútiles
    df = df.drop(columns=[
        "user_id",
        "transaction_date"
    ], errors="ignore")


    # Manejar tipo date
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])


    # 3. Asegurar target limpia
    df = df.dropna(subset=["default_flag"])

    return df