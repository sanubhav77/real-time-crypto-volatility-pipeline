"""
train.py – Train baseline and ML models for volatility spike detection

Models:
  1. Baseline: z-score threshold rule on volatility_60s
  2. ML: XGBoost classifier

Logs all parameters, metrics, and artifacts to MLflow.
Uses time-based train → validation → test splits.

Usage:
  python models/train.py --features data/processed/features.parquet
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Configuration ───────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "crypto-volatility"
ARTIFACTS_DIR = Path("models/artifacts")
REPORTS_DIR = Path("reports")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "mid_return",
    "volatility_30s", "volatility_60s", "volatility_120s",
    "mean_return_30s", "mean_return_60s", "mean_return_120s",
    "max_return_30s", "max_return_60s", "max_return_120s",
    "tick_intensity_30s", "tick_intensity_60s", "tick_intensity_120s",
    "mean_spread_30s", "mean_spread_60s", "mean_spread_120s",
    "spread_bps",
    "book_imbalance",
]

TARGET_COL = "is_spike"


def load_and_split(features_path: str):
    """Load features and perform time-based train/val/test split."""
    df = pd.read_parquet(features_path)
    df = df.sort_values("timestamp_epoch").reset_index(drop=True)

    # Drop rows without labels
    df = df.dropna(subset=[TARGET_COL, "future_volatility_60s"]).reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print(f"[INFO] Dataset: {n} rows")
    print(f"[INFO] Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"[INFO] Spike rate - Train: {train[TARGET_COL].mean():.3f} | "
          f"Val: {val[TARGET_COL].mean():.3f} | Test: {test[TARGET_COL].mean():.3f}")

    return train, val, test


def plot_pr_curve(y_true, y_scores, title, save_path):
    """Plot and save precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, color="#2196F3")
    ax.fill_between(recall, precision, alpha=0.1, color="#2196F3")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"{title}\nPR-AUC = {pr_auc:.4f}", fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # Add baseline
    baseline = y_true.mean()
    ax.axhline(baseline, color="red", linestyle="--", alpha=0.7, label=f"Random baseline: {baseline:.3f}")
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return pr_auc


def train_baseline(train, val, test):
    """
    Baseline model: z-score rule.
    If z-score of volatility_60s > threshold, predict spike.
    """
    print("\n" + "=" * 60)
    print("  BASELINE MODEL: Z-Score Rule")
    print("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="baseline_zscore"):
        # Compute z-scores based on training statistics
        train_mean = train["volatility_60s"].mean()
        train_std = train["volatility_60s"].std()

        # Score: z-score of volatility_60s (higher = more likely spike)
        def zscore(series):
            return (series - train_mean) / train_std

        val_scores = zscore(val["volatility_60s"]).values
        test_scores = zscore(test["volatility_60s"]).values

        # Find best threshold on validation set
        best_f1 = 0
        best_thresh = 1.0
        for t in np.arange(0.5, 3.0, 0.1):
            preds = (val_scores >= t).astype(int)
            f1 = f1_score(val[TARGET_COL], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        # Evaluate on test set
        test_preds = (test_scores >= best_thresh).astype(int)
        test_pr_auc = average_precision_score(test[TARGET_COL], test_scores)
        test_f1 = f1_score(test[TARGET_COL], test_preds, zero_division=0)

        print(f"  Z-score threshold: {best_thresh:.2f}")
        print(f"  Train mean volatility_60s: {train_mean:.8f}")
        print(f"  Train std volatility_60s:  {train_std:.8f}")
        print(f"  Test PR-AUC: {test_pr_auc:.4f}")
        print(f"  Test F1@threshold: {test_f1:.4f}")
        print(f"\n  Classification Report (Test):")
        print(classification_report(test[TARGET_COL], test_preds, zero_division=0))

        # Log to MLflow
        mlflow.log_param("model_type", "baseline_zscore")
        mlflow.log_param("feature_used", "volatility_60s")
        mlflow.log_param("zscore_threshold", best_thresh)
        mlflow.log_param("train_mean", train_mean)
        mlflow.log_param("train_std", train_std)
        mlflow.log_metric("pr_auc", test_pr_auc)
        mlflow.log_metric("f1_at_threshold", test_f1)
        mlflow.log_metric("val_best_f1", best_f1)

        # PR curve
        pr_path = REPORTS_DIR / "pr_curve_baseline.png"
        plot_pr_curve(test[TARGET_COL].values, test_scores,
                      "Baseline Z-Score — Precision-Recall", pr_path)
        mlflow.log_artifact(str(pr_path))

        # Save model params
        model_params = {
            "model_type": "baseline_zscore",
            "train_mean": float(train_mean),
            "train_std": float(train_std),
            "zscore_threshold": float(best_thresh),
        }
        params_path = ARTIFACTS_DIR / "baseline_params.json"
        with open(params_path, "w") as f:
            json.dump(model_params, f, indent=2)
        mlflow.log_artifact(str(params_path))

        print(f"  MLflow run logged successfully.")
        return test_pr_auc, test_f1


def train_ml_model(train, val, test):
    """
    ML model: XGBoost (or Logistic Regression fallback).
    """
    print("\n" + "=" * 60)
    print("  ML MODEL: XGBoost Classifier")
    print("=" * 60)

    # Try XGBoost, fallback to Logistic Regression
    try:
        from xgboost import XGBClassifier
        use_xgb = True
    except ImportError:
        print("[WARN] XGBoost not available, using Logistic Regression")
        use_xgb = False

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="ml_xgboost" if use_xgb else "ml_logistic_regression"):
        X_train = train[FEATURE_COLS].values
        y_train = train[TARGET_COL].values
        X_val = val[FEATURE_COLS].values
        y_val = val[TARGET_COL].values
        X_test = test[FEATURE_COLS].values
        y_test = test[TARGET_COL].values

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        if use_xgb:
            # Calculate scale_pos_weight for imbalance
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                early_stopping_rounds=20,
                random_state=42,
                use_label_encoder=False,
            )
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
            )
            model_name = "XGBoost"
        else:
            model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                C=1.0,
            )
            model.fit(X_train_scaled, y_train)
            model_name = "LogisticRegression"

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            test_proba = model.predict_proba(X_test_scaled)[:, 1]
            val_proba = model.predict_proba(X_val_scaled)[:, 1]
        else:
            test_proba = model.decision_function(X_test_scaled)
            val_proba = model.decision_function(X_val_scaled)

        # Find best threshold on validation set
        best_f1 = 0
        best_thresh = 0.5
        for t in np.arange(0.1, 0.9, 0.02):
            preds = (val_proba >= t).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        # Evaluate on test
        test_preds = (test_proba >= best_thresh).astype(int)
        test_pr_auc = average_precision_score(y_test, test_proba)
        test_f1 = f1_score(y_test, test_preds, zero_division=0)

        print(f"  Model: {model_name}")
        print(f"  Best threshold (val): {best_thresh:.2f}")
        print(f"  Test PR-AUC: {test_pr_auc:.4f}")
        print(f"  Test F1@threshold: {test_f1:.4f}")
        print(f"\n  Classification Report (Test):")
        print(classification_report(y_test, test_preds, zero_division=0))

        # Feature importance
        if use_xgb and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            imp_df = pd.DataFrame({
                "feature": FEATURE_COLS,
                "importance": importances
            }).sort_values("importance", ascending=False)
            print("  Top 10 Feature Importances:")
            for _, row in imp_df.head(10).iterrows():
                print(f"    {row['feature']:30s} {row['importance']:.4f}")

            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            imp_df_sorted = imp_df.sort_values("importance", ascending=True)
            ax.barh(imp_df_sorted["feature"], imp_df_sorted["importance"], color="#4CAF50")
            ax.set_xlabel("Importance")
            ax.set_title(f"{model_name} Feature Importance")
            plt.tight_layout()
            imp_path = REPORTS_DIR / "feature_importance.png"
            plt.savefig(imp_path, dpi=150, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(str(imp_path))

        # Log to MLflow
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("features", ",".join(FEATURE_COLS))
        mlflow.log_param("best_threshold", best_thresh)
        mlflow.log_param("train_size", len(train))
        mlflow.log_param("val_size", len(val))
        mlflow.log_param("test_size", len(test))
        mlflow.log_param("split_method", "time_based")

        if use_xgb:
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 4)
            mlflow.log_param("learning_rate", 0.05)
            mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 2))

        mlflow.log_metric("pr_auc", test_pr_auc)
        mlflow.log_metric("f1_at_threshold", test_f1)
        mlflow.log_metric("val_best_f1", best_f1)

        # PR curve
        pr_path = REPORTS_DIR / f"pr_curve_{model_name.lower()}.png"
        plot_pr_curve(y_test, test_proba,
                      f"{model_name} — Precision-Recall", pr_path)
        mlflow.log_artifact(str(pr_path))

        # Save model + scaler
        model_path = ARTIFACTS_DIR / "model.pkl"
        scaler_path = ARTIFACTS_DIR / "scaler.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(scaler_path))

        # Save metadata
        metadata = {
            "model_type": model_name,
            "feature_cols": FEATURE_COLS,
            "best_threshold": float(best_thresh),
            "test_pr_auc": float(test_pr_auc),
            "test_f1": float(test_f1),
        }
        meta_path = ARTIFACTS_DIR / "model_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(str(meta_path))

        print(f"\n  Model saved to {model_path}")
        print(f"  MLflow run logged successfully.")

        # Save test features for infer.py
        test_path = Path("data/processed/features_test.parquet")
        test.to_parquet(test_path, index=False)
        print(f"  Test set saved to {test_path}")

        return test_pr_auc, test_f1


def main():
    parser = argparse.ArgumentParser(description="Train volatility models")
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/features.parquet",
        help="Path to features Parquet file",
    )
    args = parser.parse_args()

    train, val, test = load_and_split(args.features)

    baseline_prauc, baseline_f1 = train_baseline(train, val, test)
    ml_prauc, ml_f1 = train_ml_model(train, val, test)

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<25} {'PR-AUC':>10} {'F1':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Baseline (Z-Score)':<25} {baseline_prauc:>10.4f} {baseline_f1:>10.4f}")
    print(f"  {'ML (XGBoost)':<25} {ml_prauc:>10.4f} {ml_f1:>10.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
