import joblib
import json
import argparse
import pandas as pd
from src.schema import FEATURE_COLS

def load_json(path):
    """
    parses json file to return dictionary
    """
    with open(path, 'r') as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("The file is not in the format of a dictionary")
    return data

def validate_features(d, feature_cols):
    """
    checks for missing/extra keys or non-numeric vals and returns the clean dict
    """
    extras = []
    missing = []
    for i in d.keys():
        if i not in feature_cols:
            extras.append(i)
    for j in feature_cols:
        if j not in d.keys():
            missing.append(j)
    if len(extras)!=0:
        raise ValueError(f"Input contains these extra keys: {extras}")
    if len(missing)!=0:
        raise ValueError(f"Input is missing these required keys: {missing}")
    for k, v in d.items():
        if not (isinstance(v, int) or isinstance(v, float)):
            raise TypeError(f"Non-numeric value for {k} : {v} (type {type(v)})")
    
    return d

def preprocess(d):
    """
    applies preprocessing as needed for inference: clips spi and cpi values to [0,2]
    """
    data = d.copy()
    if data["spi_early"] < 0:
        data["spi_early"] = 0
    elif data["spi_early"] > 2:
        data["spi_early"] = 2
    
    if data["cpi_early"] < 0:
        data["cpi_early"] = 0
    elif data["cpi_early"] > 2:
        data["cpi_early"] = 2
    return data

def make_feature_row(d, feature_cols):
    """
    returns a dataframe with features as instances with headers in the order of FEATURE_COLS
    """
    row = []
    for i in feature_cols:
        row.append(d[i])
    df = pd.DataFrame([row], columns=feature_cols)
    return df    
       
def main(path):
    """
    calls all the functins and returns the final prediction
    """
    d = load_json(path)
    validate_features(d, FEATURE_COLS)
    d = preprocess(d)
    X = make_feature_row(d, FEATURE_COLS)
    model = joblib.load("models/rf_delay.joblib")
    p = model.predict_proba(X)[0][1]

    return p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    p = main(args.input)
    print(f"P(delay) = {p:.3f}")







    
     
