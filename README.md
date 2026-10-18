# Data Analysis ENGIE 2026 — House Price Prediction

**Built with Shimu**

Predict house sale prices (`SalePrice`) and minimize Mean Absolute Percentage Error (MAPE).

## Best reported result
- Best public leaderboard score reported during this project: **9.11091**
- Model: five-seed CatBoost MAE ensemble
- Seeds: `42, 123, 2026, 7, 21`
- Parameters: depth `5`, learning rate `0.025`, L2 regularization `10`
- Feature engineering: total living area, porch area, quality × living area, house age, years since remodel, weighted bathrooms, rooms per 1,000 sq ft, log-area features, missingness flags.

Public leaderboard results are provisional; final/private leaderboard results can differ.

## Folder structure
```text
data-analysis-engie-2026-house-price-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/README.md
├── notebooks/ENGIE_2026_House_Price_Prediction.ipynb
├── src/features.py
├── src/train.py
├── src/make_submission.py
└── outputs/README.md
```

## Dataset
Download `train.csv` and `test.csv` from the official Kaggle competition page. For local runs, put them in `data/`. CSV data is excluded from Git by default; do not publish competition data unless the rules permit it.

## Run locally (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/train.py --train data/train.csv
python src/make_submission.py --train data/train.csv --test data/test.csv --output outputs/submission.csv
```

## Workflow
1. **EDA:** inspect shape, data types, target distribution, missing values, and numeric correlations.
2. **Feature engineering:** derive area, quality, age, bathroom, log-area and missingness features where source columns exist.
3. **Validation:** compare ExtraTrees baseline and CatBoost using a hold-out split and MAPE.
4. **Final model:** average predictions from five CatBoost seeds.
5. **Submission:** save CSV with exactly `Id,SalePrice`; verify row count and positive predictions before uploading to Kaggle.

MAPE formula:
\[
\mathrm{MAPE}=\frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i-\hat y_i}{y_i}\right|
\]

## Kaggle Notebook steps
1. Open the competition and create a Kaggle Notebook.
2. Click **Add Input** and add the competition dataset.
3. Upload/open `notebooks/ENGIE_2026_House_Price_Prediction.ipynb`.
4. Run all cells top-to-bottom. If needed, change `DATA_DIR` to the input folder that contains `train.csv` and `test.csv`.
5. Check the final cell: it saves `/kaggle/working/submission.csv`.
6. Download the CSV from the notebook Output panel, then use **Submit Predictions** on the competition page.
7. Save a notebook version and write the score/date into your experiment log.

## Experiment log
| Experiment | Model | Public score |
|---|---|---:|
| Day 18 | CatBoost | 9.49087 |
| Day 21 | Feature engineering + CatBoost MAE | 9.46597 |
| Day 22 | Five-seed CatBoost MAE ensemble | **9.11091** |

These are the scores reported during experimentation; verify them against your own Kaggle submissions page.

## Author
Built with Shimu. Attribution is useful for identifying authorship but does not itself prevent copying. Add a license only if you want to define reuse permissions.
