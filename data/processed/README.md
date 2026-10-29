# Derived modeling data

The analysis expects `model_data.csv` in this folder. The CSV is **not included** in the starter package because only the R Markdown file, not the actual dataframe, was provided for this cleanup task.

In your original RStudio session, export it with:

```r
write.csv(model_data, "model_data.csv", row.names = FALSE)
```

Then move that exported file into this `data/processed/` directory. The repository's scripts also support rebuilding it from the authorized course Parquet plus public metadata/price APIs.

**Check redistribution permission before publishing the CSV.** The underlying raw hourly file was supplied under authenticated course access. This project does not grant permission to republish protected raw data or derived records.

## Column definitions

| Column | Description |
|---|---|
| `market` | Polymarket condition ID; market identifier |
| `event_group` | Related event ID if available; otherwise unique market fallback |
| `question` | Market question from Gamma API |
| `probability` | Most recent sampled YES price at/before prediction cutoff |
| `price_dispersion` | SD of sampled YES price levels over lookback |
| `volatility` | SD of consecutive sampled YES price differences |
| `momentum` | Last minus first sampled YES price |
| `observations` | Count of historical sampled price points |
| `history_hours` | Elapsed time between earliest and latest sample in lookback |
| `minutes_since_price` | Time from last sampled price to prediction cutoff |
| `outcome` | Realized result: YES = 1, NO = 0 |
| `logit_market` | `qlogis()` of YES price clipped to [0.0001, 0.9999] |
| `volatility_pp` | `100 * volatility` |
| `momentum_pp` | `100 * momentum` |
| `dispersion_pp` | `100 * price_dispersion` |

Data context: observation cut at **2026-05-30 21:00 UTC**, 48-hour lookback; 1,094 observations after history-quality filtering in the original run. The resulting CSV contains derived analytical values, **not** raw completed trade logs.
