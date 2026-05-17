"""
eda.ipynb equivalent – Exploratory Data Analysis

Generates EDA plots and determines the volatility spike threshold.
Run: python notebooks/eda.py
Outputs plots to reports/ and prints threshold recommendation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

# ── Load Data ───────────────────────────────────────────────────────
FEATURES_PATH = Path("data/processed/features.parquet")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("[INFO] Loading features...")
df = pd.read_parquet(FEATURES_PATH)
df["dt"] = pd.to_datetime(df["timestamp"])
print(f"[INFO] Shape: {df.shape}")
print(f"[INFO] Time range: {df['dt'].min()} → {df['dt'].max()}")
print(f"[INFO] Columns: {list(df.columns)}")

# ── Basic Statistics ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  BASIC STATISTICS")
print("=" * 60)

num_cols = ["price", "midprice", "spread", "spread_bps", "mid_return",
            "volatility_30s", "volatility_60s", "volatility_120s",
            "tick_intensity_60s", "book_imbalance", "future_volatility_60s"]

stats = df[num_cols].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print(stats.to_string())

# ── Plot 1: Price over time ────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(df["dt"], df["midprice"], linewidth=0.5, color="#2196F3")
axes[0].set_ylabel("Midprice (USD)")
axes[0].set_title("BTC-USD Midprice Over Time")
axes[0].grid(True, alpha=0.3)

axes[1].plot(df["dt"], df["spread_bps"], linewidth=0.5, color="#FF9800")
axes[1].set_ylabel("Spread (bps)")
axes[1].set_title("Bid-Ask Spread")
axes[1].grid(True, alpha=0.3)

axes[2].plot(df["dt"], df["volatility_60s"], linewidth=0.5, color="#F44336")
axes[2].set_ylabel("Volatility (60s)")
axes[2].set_title("Rolling 60s Volatility of Midprice Returns")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("[PLOT] Saved: reports/eda_timeseries.png")

# ── Plot 2: Future volatility distribution + percentiles ──────────
future_vol = df["future_volatility_60s"].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(future_vol, bins=80, color="#4CAF50", alpha=0.7, edgecolor="white")
axes[0].set_xlabel("Future Volatility (60s)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Future 60s Volatility")
axes[0].axvline(future_vol.quantile(0.90), color="orange", linestyle="--", label="90th pctl")
axes[0].axvline(future_vol.quantile(0.95), color="red", linestyle="--", label="95th pctl")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Percentile plot
percentiles = np.arange(0, 100.5, 0.5)
pctl_values = [np.percentile(future_vol, p) for p in percentiles]
axes[1].plot(percentiles, pctl_values, color="#9C27B0", linewidth=2)
axes[1].set_xlabel("Percentile")
axes[1].set_ylabel("Future Volatility (60s)")
axes[1].set_title("Percentile Plot of Future Volatility")
axes[1].axhline(future_vol.quantile(0.90), color="orange", linestyle="--", alpha=0.7, label="90th pctl")
axes[1].axhline(future_vol.quantile(0.95), color="red", linestyle="--", alpha=0.7, label="95th pctl")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_volatility_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("[PLOT] Saved: reports/eda_volatility_distribution.png")

# ── Plot 3: Feature correlations ──────────────────────────────────
feature_cols = ["mid_return", "volatility_30s", "volatility_60s", "volatility_120s",
                "tick_intensity_60s", "spread_bps", "book_imbalance",
                "mean_return_60s", "max_return_60s", "future_volatility_60s"]

corr = df[feature_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(feature_cols)))
ax.set_yticks(range(len(feature_cols)))
ax.set_xticklabels([c.replace("_", "\n") for c in feature_cols], fontsize=8, rotation=45, ha="right")
ax.set_yticklabels([c.replace("_", "\n") for c in feature_cols], fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Feature Correlation Matrix")

# Add correlation values
for i in range(len(feature_cols)):
    for j in range(len(feature_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")

plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("[PLOT] Saved: reports/eda_correlation.png")

# ── Plot 4: Future volatility over time with threshold ────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["dt"], df["future_volatility_60s"], linewidth=0.5, color="#2196F3", alpha=0.6)
threshold_90 = future_vol.quantile(0.90)
threshold_95 = future_vol.quantile(0.95)
ax.axhline(threshold_90, color="orange", linestyle="--", linewidth=1.5, label=f"90th pctl = {threshold_90:.8f}")
ax.axhline(threshold_95, color="red", linestyle="--", linewidth=1.5, label=f"95th pctl = {threshold_95:.8f}")
ax.set_xlabel("Time")
ax.set_ylabel("Future Volatility (60s)")
ax.set_title("Future 60s Volatility with Spike Thresholds")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_threshold.png", dpi=150, bbox_inches="tight")
plt.close()
print("[PLOT] Saved: reports/eda_threshold.png")

# ── Threshold Analysis ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  THRESHOLD ANALYSIS")
print("=" * 60)

for pctl in [75, 80, 85, 90, 95, 99]:
    val = future_vol.quantile(pctl / 100)
    n_spikes = (future_vol >= val).sum()
    pct_spikes = n_spikes / len(future_vol) * 100
    print(f"  {pctl}th percentile: τ = {val:.8f}  →  {n_spikes} spikes ({pct_spikes:.1f}%)")

# ── Chosen threshold ──────────────────────────────────────────────
chosen_threshold = future_vol.quantile(0.90)
n_positive = (future_vol >= chosen_threshold).sum()
n_negative = (future_vol < chosen_threshold).sum()

print(f"\n  CHOSEN THRESHOLD: τ = {chosen_threshold:.8f} (90th percentile)")
print(f"  Positive labels (spike): {n_positive} ({n_positive/len(future_vol)*100:.1f}%)")
print(f"  Negative labels (no spike): {n_negative} ({n_negative/len(future_vol)*100:.1f}%)")
print(f"  Class ratio: 1:{n_negative//n_positive}")

# ── Save labeled dataset ──────────────────────────────────────────
df["is_spike"] = (df["future_volatility_60s"] >= chosen_threshold).astype(int)
df.to_parquet("data/processed/features.parquet", index=False)
print(f"\n[DONE] Added 'is_spike' label to features.parquet")
print(f"[DONE] Threshold τ = {chosen_threshold:.8f}")
