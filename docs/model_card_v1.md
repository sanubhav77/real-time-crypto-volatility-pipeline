# Model Card v1: Crypto Volatility Spike Detector

## Model Overview

| Field | Value |
|-------|-------|
| **Task** | Binary classification — detect 60-second volatility spikes in BTC-USD |
| **Models** | Baseline (z-score rule) and XGBoost classifier |
| **Primary Metric** | PR-AUC |
| **Training Data** | ~20 minutes of live BTC-USD ticker data from Coinbase (5,261 labeled rows) |
| **Framework** | scikit-learn, XGBoost, MLflow |

## Intended Use

This model is designed as a **learning exercise** for operationalizing ML pipelines. It detects short-term volatility spikes in cryptocurrency markets using streaming data. It is **not intended for live trading decisions**.

## Training Data

- **Source**: Coinbase Advanced Trade WebSocket API (ticker channel)
- **Trading pair**: BTC-USD
- **Collection period**: April 5, 2026, ~22:50–23:09 UTC (~20 minutes)
- **Total ticks**: 5,273 raw → 5,261 labeled feature rows
- **Split method**: Time-based (60% train / 20% val / 20% test)

| Split | Rows | Spike Rate |
|-------|------|------------|
| Train | 3,156 | 15.7% |
| Val | 1,052 | 0.0% |
| Test | 1,053 | 2.8% |

## Target Definition

- **Volatility proxy**: Rolling standard deviation of midprice returns over the next 60 seconds
- **Label**: `is_spike = 1` if future_volatility_60s ≥ 0.00005787 (90th percentile); else `0`
- **Class ratio**: 1:8 (overall), though distribution varies across time splits

## Features

18 input features including:
- Midprice returns and rolling volatility (30s, 60s, 120s windows)
- Bid-ask spread and spread statistics
- Tick intensity (trade frequency)
- Order book imbalance

See `docs/feature_spec.md` for full details.

## Results

| Model | PR-AUC | F1@threshold |
|-------|--------|--------------|
| Baseline (Z-Score) | 0.0164 | 0.0000 |
| XGBoost | 0.0438 | 0.0839 |

### Interpretation

Performance is low in absolute terms. Key reasons:

1. **Very limited data**: Only ~20 minutes of trading data. Volatility patterns require hours/days to capture meaningful regimes.
2. **Temporal distribution shift**: The spike rate drops from 15.7% (train) to 0.0% (val) and 2.8% (test), indicating the market entered a low-volatility regime during the later collection period.
3. **Short collection window**: The data captures a narrow price range ($68,051–$68,558), limiting the model's exposure to diverse market conditions.

Despite low absolute scores, XGBoost outperforms the baseline by 2.7x on PR-AUC, suggesting the ML model captures some signal beyond simple z-score rules.

## Top Feature Importances (XGBoost)

1. tick_intensity_60s (0.644)
2. tick_intensity_120s (0.324)
3. tick_intensity_30s (0.009)
4. mean_return_30s (0.008)

Trade intensity dominates, suggesting the model primarily uses trading frequency as a volatility predictor.

## Inference Performance

- **Speed**: 61,434x real-time (0.0034s for 1,053 rows)
- **Requirement**: < 2x real-time → ✓ PASS

## Limitations and Risks

- Model trained on a single short session; not robust to regime changes
- No cross-validation due to limited data
- Spike threshold is empirical (90th percentile) and may not generalize
- BTC-USD only; not tested on other pairs
- Validation set has 0% spike rate, preventing meaningful threshold tuning

## Ethical Considerations

- Uses public market data only; no trades are placed
- Not suitable for financial decision-making
- No personally identifiable information is used

## Recommendations for Improvement

- Collect data over multiple days/sessions to capture different volatility regimes
- Add more trading pairs for generalization
- Experiment with additional features (order flow, on-chain metrics)
- Try sequence models (LSTM, Transformer) that capture temporal patterns
- Implement online learning to adapt to regime changes
