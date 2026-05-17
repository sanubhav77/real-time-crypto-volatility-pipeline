# Handoff Note

## Selection: Selected-base

This repository provides a complete pipeline for BTC-USD volatility spike detection that can serve as the team's base model.

## Steps to Reproduce

1. Start infrastructure: `cd docker && docker compose up -d`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure `.env` from `.env.example` with your Coinbase API credentials
4. Ingest data: `python scripts/ws_ingest.py --pair BTC-USD --minutes 15`
5. Build features: `python features/featurizer.py --topic_in ticks.raw --topic_out ticks.features`
6. Train: `python models/train.py --features data/processed/features.parquet`
7. Infer: `python models/infer.py --features data/processed/features_test.parquet`

## Key Files

- `docker/compose.yaml` — Kafka + MLflow infrastructure
- `features/featurizer.py` — Feature computation pipeline
- `models/artifacts/` — Trained XGBoost model + scaler
- `docs/feature_spec.md` — Feature definitions and threshold
- `docs/model_card_v1.md` — Model performance and limitations
- `reports/model_eval.pdf` — Evaluation results
- `reports/evidently/` — Drift and quality reports

## Notes for Team

- Model trained on ~20 min of data; recommend collecting more for team project
- Tick intensity is the dominant feature; consider adding order flow features
- Threshold τ = 0.00005787 may need recalibration with more data
