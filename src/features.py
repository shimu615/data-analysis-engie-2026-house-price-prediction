"""Safe feature engineering for the ENGIE house-price dataset."""
import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def num(col):
        return pd.to_numeric(out[col], errors="coerce") if col in out.columns else None

    living = num("LivAreaSF")
    basement = num("TotalBsmtSF")
    if basement is None:
        b1, b2 = num("BsmtFinSF1"), num("BsmtFinSF2")
        if b1 is not None or b2 is not None:
            basement = (b1 if b1 is not None else 0) + (b2 if b2 is not None else 0)
    if living is not None and basement is not None:
        out["TotalLivingArea"] = living.fillna(0) + basement.fillna(0)

    porch_names = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
    porch = [num(c) for c in porch_names if c in out.columns]
    if porch:
        out["TotalPorchSF"] = pd.concat(porch, axis=1).fillna(0).sum(axis=1)

    quality = num("OverallQual")
    if quality is not None and living is not None:
        out["Quality_LivingArea"] = quality * living

    sold, built, remodel = num("YrSold"), num("YearBuilt"), num("YearRemodAdd")
    if sold is not None and built is not None:
        out["HouseAgeAtSale"] = (sold - built).clip(lower=0)
    if sold is not None and remodel is not None:
        out["YearsSinceRemodel"] = (sold - remodel).clip(lower=0)

    weights = {"FullBath": 1.0, "HalfBath": 0.5, "BsmtFullBath": 1.0, "BsmtHalfBath": 0.5}
    bath_cols = [(num(c), w) for c, w in weights.items() if c in out.columns]
    if bath_cols:
        out["TotalBathrooms"] = sum(s.fillna(0) * w for s, w in bath_cols)

    rooms = num("TotRmsAbvGrd")
    if rooms is not None and living is not None:
        out["RoomsPer1000SqFt"] = rooms / (living.replace(0, np.nan) / 1000)

    for col in ["LotArea", "LivAreaSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GarageArea"]:
        values = num(col)
        if values is not None:
            out[f"Log1p_{col}"] = np.log1p(values.clip(lower=0))

    for col in list(out.columns):
        if out[col].isna().any():
            out[f"{col}_was_missing"] = out[col].isna().astype("int8")
    return out
