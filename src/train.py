"""Fit CatBoost and report hold-out validation MAPE."""
import argparse
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from features import add_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = pd.read_csv(args.train)
    if "SalePrice" not in data:
        raise ValueError("train.csv must contain SalePrice")
    y = data["SalePrice"].astype(float)
    X = add_features(data.drop(columns=["SalePrice"]))
    cats = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for c in cats:
        X[c] = X[c].fillna("__MISSING__").astype(str)
    X = X.replace([np.inf, -np.inf], np.nan)
    cat_idx = [X.columns.get_loc(c) for c in cats]
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    model = CatBoostRegressor(loss_function="MAE", eval_metric="MAPE",
        iterations=2500, depth=5, learning_rate=0.025, l2_leaf_reg=10,
        random_seed=args.seed, verbose=250, allow_writing_files=False)
    model.fit(Xtr, ytr, eval_set=(Xv, yv), cat_features=cat_idx,
              early_stopping_rounds=200, verbose=250)
    pred = np.maximum(model.predict(Xv), 1)
    print(f"Hold-out MAPE: {mean_absolute_percentage_error(yv, pred)*100:.4f}%")
    print("Best iteration:", model.get_best_iteration())

if __name__ == "__main__":
    main()
