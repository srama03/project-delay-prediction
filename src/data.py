import pandas as pd

from src.schema import LABEL_COL, FEATURE_COLS, DROP_COLS 

def load_data(csv_path):
    """
    load and preprocess dataset:
    X: dataframe of features (in FEATURE_COLS order)
    y: series of target labels
    """
    df = pd.read_csv(csv_path)
    # drop design/leakage cols
    cols_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_drop:
        df = df.drop(columns= cols_drop)
    # clip spi and cpi
    df["spi_early"] = df["spi_early"].clip(0, 2)
    df["cpi_early"] = df["cpi_early"].clip(0, 2)
    # split into features and label
    y = df[LABEL_COL]
    X = df[FEATURE_COLS]
    return X, y		      
