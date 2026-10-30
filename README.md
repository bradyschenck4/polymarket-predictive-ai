# Can Machine Learning Beat the Market?

**STATS C161 midterm project — historical Polymarket probability forecasts**  
**Group members:** Brady Schenck, Rohan Koditala, Nason Tran, Emir Ramirez  
**Presentation:** https://docs.google.com/presentation/d/1I2ewuvhwGk0XBp3n5hy3kx9H7amUIFWPvIy-GpWUH3U/edit  
**Course snapshot:** May 30, 2026, at 2:00 p.m. Pacific (21:00 UTC)

## Research question

Can historical price behavior improve binary prediction-market outcome forecasts beyond Polymarket's own implied YES probabilities? We compare raw Polymarket probabilities with two logistic regression models, Random Forest, and XGBoost. We also assess calibration and robustness.

## Summary of the original exploratory analysis

These numbers describe **the run reported in the original R Markdown and presentation**, not results produced automatically when this repository is downloaded.

| Stage | Count |
|---|---:|
| Order-book updates in one course-provided hourly Parquet | 20,985,641 |
| Unique markets in that hour | 1,813 |
| Markets returned as closed by Gamma API | 1,473 |
| Candidate resolved YES/NO markets after date/status checks | 1,220 |
| Final modeling sample after history-quality checks | **1,094** |

The modeling sample contained **213 YES** and **881 NO** outcomes. Historical Polymarket prices had a descriptive full-sample **Brier score of 0.0771**; a constant-probability descriptive benchmark (using the full-sample YES rate) had a Brier score of **0.1568**.

### Final presentation: held-out results

These are the **reported final presentation results** on the same **209 held-out markets**, using one event-grouped split. Lower Brier scores are better.

| Method | Held-out Brier score |
|---|---:|
| **XGBoost** | **0.10526** |
| Polymarket baseline | 0.10741 |
| Random Forest | 0.10796 |
| Full logistic regression | 0.11328 |
| Price-only logistic regression | 0.11361 |

XGBoost had approximately **2.0% lower Brier score** than the raw market baseline on this test set. This modest single-split difference does not establish a consistently reproducible edge or trading profitability.

**Earlier analysis versus final presentation:** The earlier R exploration reported a Random Forest Brier score of 0.10532, a price-only logistic score of 0.10712, and a full logistic score of 0.10809. Those numbers reflect a different modeling run/configuration and should not be mixed with the final presentation comparison. The Python models use different settings from some R models; re-running scripts may produce different scores.

## Repository contents

| File | Contents |
|---|---|
| [`R/01_build_dataset.R`](R/01_build_dataset.R) | Extracts market IDs from the source Parquet, fetches Gamma metadata and CLOB price history, and engineers predictors. |
| [`R/02_analysis_models.R`](R/02_analysis_models.R) | Descriptive metrics, calibration, grouped train/test split, logistic regressions and Random Forest in R. |
| [`stats161project.py`](stats161project.py) | Legacy Colab export: exploratory order-book features and an additional Python logistic/Random Forest comparison. **Contains notebook-only syntax and embedded course-access credentials; not a standalone final-analysis script.** |
| [`xgb_model.py`](xgb_model.py) | XGBoost training-only hyperparameter search, held-out Brier evaluation, multi-seed predictions and event-grouped bootstrap. |
| [`validation.py`](validation.py) | Python refits of all five approaches, grouped bootstrap comparisons, 10-fold event-grouped CV, and calibration plots. |
| [`data/processed/README.md`](data/processed/README.md) | Derived data dictionary and data-sharing guidance. |

**Data files:** `data/processed/model_data.csv`, `train.csv`, and `test.csv` are not currently included in the public repository. The Python scripts depend on actual exported train/test CSVs, not on the 25-market pilot sample.

## Reproduce the analysis using the processed dataset (recommended)

This is the simplest option for teammates. In RStudio, set the working directory to the cloned repository root and run:

```r
install.packages(c("dplyr", "ggplot2", "ranger"))
source("R/02_analysis_models.R")
```

**First**, place `model_data.csv` in `data/processed/`. If using `model_data.csv` exported from our original RStudio project, retain its row order to reproduce the seeded event-group split as closely as possible. The script generates:

- `results/descriptive_baselines.csv`
- `results/calibration_bins.csv` and `results/calibration_plot.png`
- `results/train.csv` and `results/test_predictions.csv`
- `results/model_comparison.csv` and `results/model_brier_comparison.png`

The script re-creates the derived predictor transformations and uses `set.seed(42)`, an 80/20 split **by event group**, logistic regression, and `ranger` Random Forest with 500 trees and minimum node size 20.

The **original** analysis was run in RStudio with a particular package environment and historical API data. Exact numeric results may vary if source data, row order, event grouping, or package versions change.

## Rebuild the modeling data from source (optional)

Only attempt this if you have lawful access to the course-provided Parquet file. Do **not** put database usernames, passwords, secret URLs, or the restricted raw archive in a public repository.

```r
install.packages(c("arrow", "dplyr", "jsonlite", "ggplot2", "ranger"))
source("R/01_build_dataset.R")
```

The source script looks for `data/raw/events-2026-05-30T21.parquet` by default. When running with `Rscript`, you can pass a different local path:

```sh
Rscript R/01_build_dataset.R /absolute/path/to/events-2026-05-30T21.parquet
Rscript R/02_analysis_models.R
```

It extracts market IDs from the local Parquet, retrieves **closed** market metadata using [Gamma](https://gamma-api.polymarket.com/), fetches historical YES prices using the [CLOB API](https://clob.polymarket.com/), and builds features at the fixed cutoff time. Network access is required for rebuilding. Cached API responses are kept under `data/cache/`, which is excluded from Git. Source APIs can have rate limits or change over time.

## Running the Python models

Install required packages:

```bash
python -m pip install numpy pandas scikit-learn xgboost matplotlib
```

Place your actual event-grouped `train.csv` and `test.csv` in `data/processed/` and execute from the repository root:

```bash
python xgb_model.py data/processed
python validation.py data/processed
```

Both scripts require `probability`, `volatility`, `momentum`, `price_dispersion`, `outcome`, and `event_group` in the CSVs. `validation.py` additionally uses the transformed columns `logit_market`, `volatility_pp`, `momentum_pp`, and `dispersion_pp`. To print comparisons against prior model predictions, `xgb_model.py` requires `pred_price`, `pred_full`, and `pred_rf` in `test.csv`. These CSVs must come from the appropriate modeling split.

### XGBoost tuning (`xgb_model.py`)

- Uses **5-fold event-grouped cross-validation on the training data only** to choose among 16 hyperparameter combinations: number of trees (100/300), max depth (2/3), learning rate (0.03/0.1), and minimum child weight (5/10).
- Fits the selected model with **10 random seeds** and averages the held-out YES-probability forecasts.
- Compares its Brier score against Polymarket and existing prediction columns; uses **5,000 event-grouped bootstrap resamples** for a paired XGBoost-minus-market Brier difference confidence interval.
- Saves `test_with_xgb.csv` in the working directory.

### Model validation (`validation.py`)

- Re-fits the market baseline, both logistic regressions, Random Forest, and a **fixed-parameter** XGBoost model using Python.
- Evaluates held-out Brier scores and 95% event-grouped bootstrap intervals for paired differences versus Polymarket.
- Runs a **10-fold event-grouped cross-validation** analysis across all available markets, and saves `calibration.png`.
- This is an additional grouped robustness analysis, **not a time-forward backtest**. It is also not the same as the train-only hyperparameter tuning in `xgb_model.py`. Python refit results may differ from the slide scores.

### Exploratory Python (`stats161project.py`)

This is a Colab-exported exploration, not an end-to-end reproducible command-line script. It includes `!pip` notebook syntax, a distinct October 9 order-book experiment involving prediction of the next quoted price movement, and a separate four-model comparison using `train.csv`/`test.csv`. Order-book updates should not be described as executed trades. The file contains hard-coded historical classroom credentials and should be **sanitized immediately**; expiration does not make publishing credentials a good practice. The owner of those credentials should revoke or rotate them if possible. Removing them from a later commit alone does not erase earlier Git history.

For reproducible final-project modeling, use the R pipeline plus `xgb_model.py` and `validation.py` with the correct exported CSVs.

## Methodology

Each observation represents one YES/NO market, with features calculated using information available **up to May 30, 2026, 21:00 UTC**:

| Feature | Description |
|---|---|
| `probability` | Most recent sampled YES price before prediction time |
| `momentum` | Last minus first sampled YES price in the 48-hour lookback |
| `volatility` | SD of consecutive sampled price changes |
| `price_dispersion` | SD of sampled price levels |
| `logit_market` | Clipped log-odds transform of `probability` |
| `*_pp` | Price-based features multiplied by 100 (percentage-point scale) |
| `outcome` | 1 = resolved YES, 0 = resolved NO |

Data-quality rules: at least 40 hours of sampled history, a latest sampled price no more than 10 minutes before the cutoff, finite predictors, and a valid binary outcome label. Market metadata must identify the market as resolved with final-looking 0/1 outcome prices and a closing time after the forecast cutoff.

The entire 1,094-market dataset is **not** used to train and evaluate the same model: we reserve event groups for testing, and report model performance using only the held-out markets.

## Important limitations

1. **Event timing** — A market can become effectively decided before its technical closing time. The closing-time check does **not** prove the outcome was unknown at the forecast cutoff; event-level verification is still needed.
2. **Single cutoff** — All feature windows end on the same May 30 timestamp; the experiment is not a time-forward backtest.
3. **Related contracts** — Event IDs are used for grouped splitting when present; markets missing those IDs fall back to market-specific groups, so dependence can remain.
4. **Small model differences** — One random grouped split cannot establish consistent outperformance. Repeat splits and report uncertainty before making strong predictive claims.
5. **Market activity vs. trades** — The Parquet contains order-book changes. The quoted `size` column should **not** be summed as executed trading volume.
6. **Profitability** — A lower Brier score does not by itself imply a profitable trading strategy after transaction costs and execution constraints.

## Public data / redistribution warning

The original 21-million-row Parquet file was supplied via **time-limited, authenticated course access**. Its redistribution rights are **not established by this repository**. We intentionally exclude that raw dataset, cached metadata, and any access credentials from Git. The modeling CSV is a much smaller **derived** file, but even a derived file may be covered by the original terms; **confirm with the instructor or data provider before committing `model_data.csv` publicly**.

Your course requires the data and code to be publicly viewable. To fulfill the data requirement responsibly, either (a) obtain permission to share the derived `data/processed/model_data.csv`, or (b) use a public-data source and rebuild the dataset. Do not substitute a fabricated CSV for actual results.

## References

- Course-provided Polymarket hourly Parquet (access-controlled; not redistributed)
- Polymarket [Gamma API](https://docs.polymarket.com/) market metadata
- Polymarket [CLOB API](https://docs.polymarket.com/) price history
- Kalshi Research (2026), *Calibration in Prediction Markets — Theory and Evidence* (background reading; check course citation format)
