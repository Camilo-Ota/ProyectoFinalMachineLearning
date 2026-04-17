import pandas as pd
import os

# CARGAR DATASET RAW
def load_raw_data(path="data/raw/Buy_Now_Pay_Later_BNPL_CreditRisk_Dataset.csv"):

    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo en: {path}")

    df = pd.read_csv(path)

    print(f"Dataset cargado correctamente. Shape: {df.shape}")

    return df