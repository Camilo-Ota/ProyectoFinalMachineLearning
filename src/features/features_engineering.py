import pandas as pd
import numpy as np


def feature_engineering(df):

    df_clean = df.copy()

    # ELIMINAR COLUMNAS
    df_clean = df_clean.drop(columns=[
        'user_id',
        'product_category',
        'location',
        'transaction_date'
    ])


    # AGRUPAR

    df_clean['age_group'] = pd.cut(
        df_clean['age'],
        bins=[18, 25, 35, 50, 60],
        labels=['Joven', 'Joven_Adulto', 'Adulto', 'Mayor']
    )

    df_clean['income_group'] = pd.cut(
        df_clean['monthly_income'],
        bins=[0, 20000, 50000, 100000],
        labels=['Bajo', 'Medio', 'Alto']
    )

    df_clean['credit_group'] = pd.cut(
        df_clean['credit_score'],
        bins=[300, 580, 670, 740, 850],
        labels=['Malo', 'Regular', 'Bueno', 'Excelente']
    )

    df_clean['dti_group'] = pd.cut(
        df_clean['debt_to_income_ratio'],
        bins=[0, 0.2, 0.4, 1],
        labels=['Bajo', 'Medio', 'Alto']
    )

    df_clean['purchase_group'] = pd.cut(
        df_clean['purchase_amount'],
        bins=[0, 1000, 5000, 20000],
        labels=['Bajo', 'Medio', 'Alto']
    )


    # ELIMINAR NUMÉRICAS ORIGINALES

    df_clean = df_clean.drop(columns=[
        'age',
        'monthly_income',
        'credit_score',
        'debt_to_income_ratio',
        'purchase_amount'
    ])


    #DUMMIES

    df_model = pd.get_dummies(df_clean, drop_first=True)

    return df_model