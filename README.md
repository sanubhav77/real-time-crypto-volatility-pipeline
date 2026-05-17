# Crypto Volatility Spike Detection

Real-time pipeline for detecting short-term volatility spikes in BTC-USD using streaming data from Coinbase.

## Overview

This project builds an end-to-end ML pipeline that:
1. **Streams** live ticker data from Coinbase via WebSocket → Kafka
2. **Engineers features** (rolling volatility, spread, tick intensity) from the stream
3. **Trains models** (z-score baseline + XGBoost) to predict 60-second volatility spikes
4. **Tracks experiments** with MLflow and monitors data quality with drift reports

## Quick Start

```bash
# 1. Start infrastructure
cd docker && docker compose up -d

# 2. Install dependencies
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 3. Ingest 15 minutes of live data
python scripts/ws_ingest.py --pair BTC-USD --minutes 15

# 4. Verify Kafka messages
python scripts/kafka_consume_check.py --topic ticks.raw --min 100

# 5. Build features
python features/featurizer.py --topic_in ticks.raw --topic_out ticks.features

# 6. Replay to verify consistency
python scripts/replay.py --raw data/raw/*.ndjson --out data/processed/features_replay.parquet

# 7. Run EDA
python notebooks/eda.py

# 8. Generate drift reports
python scripts/generate_evidently_report.py

# 9. Train models
python models/train.py --features data/processed/features.parquet

# 10. Run inference
python models/infer.py --features data/processed/features_test.parquet
```

## Project Structure

```
├── data/raw/               Raw NDJSON tick data
├── data/processed/         Feature Parquet files
├── features/               Featurizer (Kafka consumer)
├── models/                 Training, inference, saved artifacts
├── notebooks/              EDA
├── reports/                Evaluation plots + Evidently drift reports
├── scripts/                Ingest, replay, Kafka check, Evidently
├── docker/                 Compose + Dockerfile
├── docs/                   Scoping brief, feature spec, model card
├── handoff/                Team handoff files
├── config.yaml             Project configuration
├── requirements.txt        Python dependencies
└── README.md               This file
```

## Tools

- **Python 3.10+**, Kafka (KRaft), Docker Compose
- **MLflow** for experiment tracking
- **Evidently** (scipy-based) for drift/quality monitoring
- **XGBoost** for ML model

## Results

| Model | PR-AUC | F1 |
|-------|--------|----|
| Baseline (Z-Score) | 0.016 | 0.000 |
| XGBoost | 0.044 | 0.084 |

See `docs/model_card_v1.md` for full analysis.

## Configuration

API credentials go in `.env` (never committed):
```
COINBASE_API_KEY=your_key
COINBASE_API_SECRET=your_secret
```
