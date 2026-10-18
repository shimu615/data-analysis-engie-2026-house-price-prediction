"""Train a five-seed CatBoost ensemble and write a Kaggle submission."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from features import add_features

SEEDS = [42, 123, 2026, 7, 21]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="outputs/submission.csv")
    args = parser.parse_args()
    train, test = pd.read_csv(args.train), pd.read_csv(args.test)
    if "SalePrice" not in train.columns or "Id" not in test.columns:
        raise ValueError("Expected SalePrice in train.csv and Id in test.csv")
    y = train["SalePrice"].astype(float)
    X, Xt = add_features(train.drop(columns=["SalePrice"])), add_features(test)
    X, Xt = X.align(Xt, join="outer", axis=1)
    cats = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for c in cats:
        X[c] = X[c].fillna("__MISSING__").astype(str)
        Xt[c] = Xt[c].fillna("__MISSING__").astype(str)
    X, Xt = X.replace([np.inf, -np.inf], np.nan), Xt.replace([np.inf, -np.inf], np.nan)
    cat_idx = [X.columns.get_loc(c) for c in cats]
    preds = []
    # Approximation of the reported winning configuration. For exact replication,
    # replace ITERATIONS with the median best iteration from the original Day 22 run.
    ITERATIONS = 2500
    for seed in SEEDS:
        model = CatBoostRegressor(loss_function="MAE", iterations=ITERATIONS,
            depth=5, learning_rate=0.025, l2_leaf_reg=10, random_seed=seed,
            verbose=250, allow_writing_files=False)
        model.fit(X, y, cat_features=cat_idx, verbose=250)
        preds.append(model.predict(Xt))
    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": np.maximum(np.mean(preds, axis=0), 1)})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)
    print("Saved:", out.resolve(), "| rows:", len(submission),
          "| missing:", submission["SalePrice"].isna().sum())

if __name__ == "__main__":
    main()
