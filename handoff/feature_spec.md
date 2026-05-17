# Feature Specification

## Target Definition

| Component | Value |
|-----------|-------|
| **Target horizon** | 60 seconds |
| **Volatility proxy** | Rolling standard deviation of midprice returns over the next 60s window |
| **Label definition** | `1` if σ_future ≥ τ; else `0` |
| **Chosen threshold τ** | 0.00005787 (90th percentile of observed future volatility) |

## Threshold Justification

The threshold was selected at the **90th percentile** of the future 60-second volatility distribution based on EDA analysis of 5,265 feature rows. This yields:

- **527 positive labels (spike)** — 10.0% of data
- **4,734 negative labels (no spike)** — 90.0% of data
- **Class ratio:** 1:8

The 90th percentile was chosen as a balance between:
- Having enough positive samples for the model to learn meaningful patterns
- Defining "spike" as genuinely unusual volatility (top 10%)
- Avoiding extreme imbalance that would occur at higher percentiles (95th → 5%, 99th → 1%)

Percentile analysis from EDA:

| Percentile | τ Value | Spike Count | Spike % |
|------------|---------|-------------|---------|
| 75th | 0.00004986 | 1,316 | 25.0% |
| 80th | 0.00005122 | 1,053 | 20.0% |
| 85th | 0.00005524 | 791 | 15.0% |
| **90th** | **0.00005787** | **527** | **10.0%** |
| 95th | 0.00006794 | 265 | 5.0% |
| 99th | 0.00007413 | 53 | 1.0% |

## Feature Descriptions

### Raw Tick Fields
- **price**: Last trade price from Coinbase ticker
- **midprice**: (best_bid + best_ask) / 2
- **best_bid / best_ask**: Current top-of-book prices
- **spread**: best_ask - best_bid (in USD)
- **spread_bps**: Spread in basis points = (spread / midprice) × 10,000

### Computed Features (per window: 30s, 60s, 120s)
- **mid_return**: Percentage change between consecutive midprices
- **volatility_{W}s**: Rolling std of midprice returns within the past W seconds
- **mean_return_{W}s**: Rolling mean of midprice returns within the past W seconds
- **max_return_{W}s**: Max absolute midprice return within the past W seconds
- **tick_intensity_{W}s**: Number of ticks per second in the past W seconds
- **mean_spread_{W}s**: Average bid-ask spread within the past W seconds

### Order Book Feature
- **book_imbalance**: (bid_quantity - ask_quantity) / (bid_quantity + ask_quantity)

### Target
- **future_volatility_60s**: Rolling std of midprice returns in the NEXT 60 seconds
- **is_spike**: Binary label, 1 if future_volatility_60s ≥ 0.00005787

## Data Source

- **API**: Coinbase Advanced Trade WebSocket (ticker channel)
- **Trading pair**: BTC-USD
- **Collection period**: ~20 minutes of live data
- **Total ticks**: 5,273 raw ticks → 5,265 feature rows

## Replay Consistency

Features generated via live Kafka consumer and offline replay from NDJSON both produce identical results: 5,265 rows × 27 columns (before label addition).
