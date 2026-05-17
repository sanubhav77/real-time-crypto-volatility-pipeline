"""
infer.py – Score features using a trained model

Loads the trained model and scaler from artifacts, scores input features,
and outputs predictions. Measures inference time to verify < 2x real-time.

Usage:
  python models/infer.py --features data/processed/features_test.parquet
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path("models/artifacts")


def load_model():
    """Load trained model, scaler, and metadata."""
    model_path = ARTIFACTS_DIR / "model.pkl"
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    meta_path = ARTIFACTS_DIR / "model_metadata.json"

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    print(f"[INFO] Loaded model: {metadata['model_type']}")
    print(f"[INFO] Features: {len(metadata['feature_cols'])} columns")
    print(f"[INFO] Threshold: {metadata['best_threshold']}")

    return model, scaler, metadata


def main():
    parser = argparse.ArgumentParser(description="Score features with trained model")
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/features_test.parquet",
        help="Path to features Parquet file to score",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/predictions.parquet",
        help="Path to save predictions",
    )
    args = parser.parse_args()

    # Load model
    model, scaler, metadata = load_model()
    feature_cols = metadata["feature_cols"]
    threshold = metadata["best_threshold"]

    # Load features
    print(f"[INFO] Loading features from: {args.features}")
    df = pd.read_parquet(args.features)
    print(f"[INFO] Input shape: {df.shape}")

    X = df[feature_cols].values

    # ── Inference with timing ───────────────────────────────────────
    print(f"\n[INFO] Running inference on {len(X)} rows...")

    start_time = time.time()
    X_scaled = scaler.transform(X)
    probas = model.predict_proba(X_scaled)[:, 1]
    preds = (probas >= threshold).astype(int)
    elapsed = time.time() - start_time

    # ── Timing analysis ─────────────────────────────────────────────
    # Data spans some time window; check if inference is < 2x real-time
    if "timestamp_epoch" in df.columns:
        data_duration = df["timestamp_epoch"].max() - df["timestamp_epoch"].min()
    else:
        data_duration = len(df)  # fallback

    print(f"\n{'=' * 60}")
    print(f"  INFERENCE RESULTS")
    print(f"{'=' * 60}")
    print(f"  Rows scored:        {len(preds)}")
    print(f"  Predicted spikes:   {preds.sum()} ({preds.mean()*100:.1f}%)")
    print(f"  Inference time:     {elapsed:.4f}s")
    print(f"  Data duration:      {data_duration:.1f}s")
    print(f"  Speedup:            {data_duration/elapsed:.1f}x real-time")

    if elapsed < 2 * data_duration:
        print(f"  ✓ PASS — Inference is {data_duration/elapsed:.1f}x faster than real-time (< 2x required)")
    else:
        print(f"  ✗ FAIL — Inference too slow")

    # ── Save predictions ────────────────────────────────────────────
    df["predicted_spike"] = preds
    df["spike_probability"] = probas

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n[DONE] Predictions saved to: {out_path}")

    # Also save a CSV sample for inspection
    sample_path = out_path.parent / "predictions_sample.csv"
    sample_cols = ["timestamp", "price", "volatility_60s", "future_volatility_60s",
                   "is_spike", "spike_probability", "predicted_spike"]
    sample_cols = [c for c in sample_cols if c in df.columns]
    df[sample_cols].head(50).to_csv(sample_path, index=False)
    print(f"[DONE] Sample predictions saved to: {sample_path}")


if __name__ == "__main__":
    main()
