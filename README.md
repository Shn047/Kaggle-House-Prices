# House Prices: Advanced Regression Techniques (Kaggle)

Predicting house sale prices in Ames, Iowa from 79 property features, using a reusable scikit-learn pipeline and a stacked ensemble of 7 Optuna-tuned models.

**Result: rank #419 (top 8%) on the Kaggle leaderboard, RMSE 0.12119**, up from rank #734 (RMSE 0.12603) with our best single model in the previous phase.

Team project (4 students) for the Data Science course at KMUTT, Bangkok.
**My role:** coordinated the team and was responsible for the modelling: feature engineering, the scikit-learn pipeline, Optuna tuning and the final stacked ensemble.

---

## Data

- Kaggle *House Prices: Advanced Regression Techniques*
- 1,460 training homes and 1,459 test homes, 79 features each
- Target: `SalePrice`, modelled as `log(SalePrice)` because prices are right-skewed
- Leaderboard metric: RMSE between the log of predicted and actual prices

## Approach

The project ran in three phases:

1. **Exploration:** distributions, missing data and the strongest relationships with price (overall quality, living area, garage capacity, neighbourhood, exterior quality).
2. **Single models:** XGBoost, CatBoost, LightGBM and TabPFN, tuned with Optuna. Best: XGBoost, rank #734.
3. **Stacked ensemble** (this notebook): rank #419.

### Feature engineering
- Size and age: `TotalSF`, `TotalBath`, `PropertyAge`
- Yes/no flags: `HasPool`, `HasBasement`, `HasFireplace`, `Has2ndFloor`, `HasGarage`, `HasRemodeled`
- Categorical views of numbers: `YrSold`, `MoSold`, `YearBuilt`, `MSSubClass`
- Quality interactions: `OverallQual × OverallCond`, `OverallQual × GrLivArea`

### Preprocessing (one scikit-learn `Pipeline`)
- Feature engineering wrapped in a `FunctionTransformer`
- Categorical: fill missing with `"None"` (usually means the home has no such feature), then one-hot encode
- Numerical: fill missing with the mean, then standardise

Keeping every step in one pipeline means training, validation and the final submission all use exactly the same transformations.

### Models
Each model was tuned with Optuna (about 50 trials, 5-fold cross-validated RMSE on log price):

| Model | Best CV RMSE |
|---|---|
| XGBoost | 0.1172 |
| KernelRidge | 0.1211 |
| LightGBM | 0.1255 |
| ExtraTrees | 0.1304 |
| ElasticNet | 0.1338 |
| Lasso | 0.1344 |
| Ridge | 0.1369 |

The seven tuned models are combined with scikit-learn's `StackingRegressor`, using `RidgeCV` as the meta-model (5-fold). CatBoost was also tried, but its tuning was stopped before it finished, so it was left out of the final stack.

## Results

| Phase | Model | Rank | RMSE |
|---|---|---|---|
| 2 | Tuned XGBoost (single model) | #734 | 0.12603 |
| Final | Stacked ensemble of 7 models | **#419 (top 8%)** | **0.12119** |

## Known issues

Found when reviewing the notebook after the competition. The code is left as submitted so it matches the leaderboard result; each issue is explained at the end of the notebook.

- **Log transform mismatch:** training uses `np.log`, but predictions are reversed with `np.expm1`, so each prediction is $1 too low (negligible at these prices). Fix: `np.exp`.
- **High-null columns not dropped:** `df.drop(...)` was not assigned back, so those columns were imputed and encoded instead.
- **Tree count mismatch:** LightGBM and XGBoost were tuned with 2,000 trees but used 1,000 in the final stack, so they are probably under-trained.
- **Slightly optimistic CV scores:** preprocessing was fitted before cross-validation instead of inside it.
- **Not fully reproducible:** Optuna was run without a fixed seed.

## What I learned
- A single reproducible pipeline prevents train/test mismatches and makes experiments comparable.
- Near the top of a leaderboard, small RMSE gains move you a long way: under 0.005 took us from #734 to #419.
- Stacking improves accuracy but is hard to explain; in a business setting, a simpler model can be the better choice.

## Files
- `ensemble-with-multiple-models.ipynb`: feature engineering, pipeline, Optuna tuning, stacking and submission (outputs cleared; recorded results are at the top of the notebook)

## Tools
Python, pandas, NumPy, scikit-learn, Optuna, LightGBM, XGBoost
