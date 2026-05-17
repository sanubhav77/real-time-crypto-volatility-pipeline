# Generative AI Usage Appendix

This document records how generative AI tools were used during this project, per the assignment requirements.

## Tool Used

- **Tool**: Claude (Anthropic)
- **Purpose**: Code review, debugging assistance, syntax help, and documentation formatting

## Usage Log

### 1. Docker Compose Setup
- **Prompt (summary)**: "Review my Docker Compose config for Kafka KRaft and MLflow — MLflow isn't binding to the right address"
- **Used in**: `docker/compose.yaml`
- **Verification**: Tested with `docker compose up -d` and `docker compose ps`. Identified MLflow was binding to 127.0.0.1 instead of 0.0.0.0 and applied a fix using `sed` to patch the server config at startup.

### 2. WebSocket Ingestor
- **Prompt (summary)**: "Help me debug my Coinbase WebSocket connection — ticker subscription isn't going through"
- **Used in**: `scripts/ws_ingest.py`
- **Verification**: Identified that Coinbase CDP API keys require JWT (ES256) authentication instead of HMAC. Updated the auth logic accordingly. Tested with live data — confirmed 5,273 ticks ingested over 15 minutes.

### 3. Kafka Consumer Check
- **Prompt (summary)**: "Check my Kafka consumer script for reading back messages from ticks.raw — want to make sure offset handling is correct"
- **Used in**: `scripts/kafka_consume_check.py`
- **Verification**: Ran and confirmed 5,273 messages with PASS status.

### 4. Featurizer
- **Prompt (summary)**: "Review my featurizer logic for computing rolling windowed features — need help structuring the deque-based buffer"
- **Used in**: `features/featurizer.py`
- **Verification**: Verified 5,265 feature rows produced. Confirmed replay consistency (identical output from live and replay modes).

### 5. Replay Script
- **Prompt (summary)**: "Help me wire up the replay script to reuse the same FeatureEngine class from the featurizer"
- **Used in**: `scripts/replay.py`
- **Verification**: Output matches live featurizer: 5,265 rows, 27 columns.

### 6. EDA Script
- **Prompt (summary)**: "Help me format the percentile plot and threshold analysis table for the EDA"
- **Used in**: `notebooks/eda.py`
- **Verification**: Reviewed plots and threshold analysis. Selected 90th percentile (τ = 0.00005787) based on class balance considerations.

### 7. Drift & Quality Report
- **Prompt (summary)**: "Evidently is crashing with Python 3.14 — help me build an equivalent drift report using scipy"
- **Used in**: `scripts/generate_evidently_report.py`
- **Verification**: Implemented KS tests and Wasserstein distance as alternatives. Reviewed HTML output for correctness.

### 8. Training Script
- **Prompt (summary)**: "Review my train.py — need help with MLflow logging syntax and PR curve plotting"
- **Used in**: `models/train.py`
- **Verification**: Both runs visible in MLflow UI. Reviewed PR-AUC, F1, and classification reports. Confirmed time-based splits.

### 9. Inference Script
- **Prompt (summary)**: "Help me add timing measurement to my inference script to check the 2x real-time requirement"
- **Used in**: `models/infer.py`
- **Verification**: Inference completes at 61,434x real-time (PASS). Predictions saved to Parquet.

### 10. Documentation
- **Prompt (summary)**: "Help me structure the scoping brief and model card — want to make sure I cover all required sections"
- **Used in**: `docs/scoping_brief.pdf`, `docs/feature_spec.md`, `docs/model_card_v1.md`
- **Verification**: Reviewed all documents for accuracy against actual results and assignment requirements.

## General Notes

- AI was primarily used for debugging, code review, and formatting assistance
- All code was tested and validated before inclusion
- Domain decisions (threshold selection, feature choices, model interpretation) were made based on EDA results and assignment requirements
- Key debugging sessions included: MLflow network binding, JWT authentication for Coinbase CDP keys, and Evidently Python 3.14 incompatibility
