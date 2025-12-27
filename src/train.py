# imports:
# libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
# internal dependencies
from src.data import load_data

params  = {
    "n_estimators" : 300,
    "max_depth" : 5,
    "min_samples_leaf" : 5,
    "max_features" : None,
    "random_state" : 42
} 

def split(X, y, seed=42):
    """
    takes the feature matrix and targets
    do a two-step stratified split:
    first split: train vs temp (70/30)
    second split: temp → val/test (15/15 overall)
    return the split datasets X_train, X_val, X_test, y_train, y_val, y_test
    """

    # stratified split 1: df-> train and temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    # stratified split 2: temp-> val and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_rf(X_train, y_train, params):
    """
    train random forest on only the training data
    return the fit model
    """
    rf = RandomForestClassifier(n_estimators= params["n_estimators"], max_depth=params["max_depth"],min_samples_leaf=params["min_samples_leaf"], max_features=params["max_features"],random_state= params["random_state"]).fit(X_train, y_train)
    return rf

def evaluate_model(model, X, y):
    """
    Takes model, feature matrix, and labels
    Returns a dict of accuracy, f1 score, and roc-auc score
    """
    # calculate hard preds and pred probs
    y_preds = model.predict(X)
    y_probs = model.predict_proba(X)[:, 1]
    # compute metrics
    acc = accuracy_score(y, y_preds)
    f1 = f1_score(y, y_preds)
    roc_auc = roc_auc_score(y, y_probs)
    metrics = {
        "accuracy" : acc,
        "f1" : f1,
        "roc_auc" : roc_auc
    }
    return metrics

def run_training(csv_path, params):
    """
    calls load → split → train → evaluate
    returns (model, val_metrics, test_metrics)
    """
    # load
    X, y = load_data(csv_path)
    # split
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
    # fit model
    rf = train_rf(X_train, y_train, params)
    # eval
    val_metrics = evaluate_model(rf, X_val, y_val)
    test_metrics = evaluate_model(rf, X_test, y_test)

    return (rf, val_metrics, test_metrics)

# main function:

if __name__ == "__main__":
    csv_path = "data/rg30_set1.csv"
    run_training(csv_path, params) 








    
