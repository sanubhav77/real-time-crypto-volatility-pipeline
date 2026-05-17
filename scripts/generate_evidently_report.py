"""
generate_evidently_report.py – Data Quality & Drift Report

Since Evidently has compatibility issues with Python 3.14,
this script generates equivalent drift and data quality reports
using scipy stats tests and saves them as HTML.

Compares early vs late windows of collected data.

Usage:
  python scripts/generate_evidently_report.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime

FEATURES_PATH = Path("data/processed/features.parquet")
REPORTS_DIR = Path("reports/evidently")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("[INFO] Loading features...")
df = pd.read_parquet(FEATURES_PATH)
df = df.sort_values("timestamp_epoch").reset_index(drop=True)
print(f"[INFO] Shape: {df.shape}")

# ── Split into early and late windows ───────────────────────────────
midpoint = len(df) // 2
df_early = df.iloc[:midpoint].copy()
df_late = df.iloc[midpoint:].copy()

print(f"[INFO] Early window: {len(df_early)} rows")
print(f"[INFO] Late window:  {len(df_late)} rows")

feature_cols = [
    "price", "midprice", "spread", "spread_bps", "mid_return",
    "volatility_30s", "volatility_60s", "volatility_120s",
    "mean_return_30s", "mean_return_60s", "mean_return_120s",
    "max_return_30s", "max_return_60s", "max_return_120s",
    "tick_intensity_30s", "tick_intensity_60s", "tick_intensity_120s",
    "mean_spread_30s", "mean_spread_60s", "mean_spread_120s",
    "book_imbalance",
]
feature_cols = [c for c in feature_cols if c in df.columns]


# ══════════════════════════════════════════════════════════════════
#  DATA DRIFT REPORT
# ══════════════════════════════════════════════════════════════════
drift_results = []
for col in feature_cols:
    early_vals = df_early[col].dropna().values
    late_vals = df_late[col].dropna().values
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(early_vals, late_vals)
    
    # Wasserstein distance
    wasserstein = stats.wasserstein_distance(early_vals, late_vals)
    
    drift_detected = ks_pval < 0.05
    
    drift_results.append({
        "feature": col,
        "early_mean": np.mean(early_vals),
        "late_mean": np.mean(late_vals),
        "early_std": np.std(early_vals),
        "late_std": np.std(late_vals),
        "ks_statistic": ks_stat,
        "ks_p_value": ks_pval,
        "wasserstein_distance": wasserstein,
        "drift_detected": drift_detected,
    })

drift_df = pd.DataFrame(drift_results)
n_drifted = drift_df["drift_detected"].sum()

print(f"\n[DRIFT] {n_drifted}/{len(feature_cols)} features show drift (KS test, p<0.05)")

# ══════════════════════════════════════════════════════════════════
#  DATA QUALITY REPORT
# ══════════════════════════════════════════════════════════════════
quality_results = []
for col in feature_cols:
    vals = df[col]
    quality_results.append({
        "feature": col,
        "count": len(vals),
        "missing": vals.isna().sum(),
        "missing_pct": vals.isna().mean() * 100,
        "zeros": (vals == 0).sum(),
        "zeros_pct": (vals == 0).mean() * 100,
        "mean": vals.mean(),
        "std": vals.std(),
        "min": vals.min(),
        "max": vals.max(),
        "q25": vals.quantile(0.25),
        "q50": vals.quantile(0.50),
        "q75": vals.quantile(0.75),
        "iqr": vals.quantile(0.75) - vals.quantile(0.25),
    })

quality_df = pd.DataFrame(quality_results)


# ══════════════════════════════════════════════════════════════════
#  GENERATE HTML REPORTS
# ══════════════════════════════════════════════════════════════════

def generate_drift_html(drift_df, n_early, n_late):
    rows_html = ""
    for _, r in drift_df.iterrows():
        color = "#ffcccc" if r["drift_detected"] else "#ccffcc"
        status = "⚠ DRIFT" if r["drift_detected"] else "✓ OK"
        rows_html += f"""
        <tr style="background-color: {color};">
            <td>{r['feature']}</td>
            <td>{r['early_mean']:.8f}</td>
            <td>{r['late_mean']:.8f}</td>
            <td>{r['early_std']:.8f}</td>
            <td>{r['late_std']:.8f}</td>
            <td>{r['ks_statistic']:.4f}</td>
            <td>{r['ks_p_value']:.6f}</td>
            <td>{r['wasserstein_distance']:.8f}</td>
            <td><b>{status}</b></td>
        </tr>"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Data Drift Report – Early vs Late Window</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #fafafa; }}
        h1 {{ color: #1a1a2e; }}
        h2 {{ color: #16213e; margin-top: 30px; }}
        .summary {{ background: #fff; padding: 20px; border-radius: 8px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th {{ background: #1a1a2e; color: white; padding: 10px 8px; text-align: left; font-size: 13px; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; font-size: 12px; }}
        .metric {{ font-size: 28px; font-weight: bold; color: #1a1a2e; }}
        .label {{ font-size: 14px; color: #666; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .card {{ background: #fff; padding: 20px; border-radius: 8px; text-align: center;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h1>📊 Data Drift Report</h1>
    <p>Comparing <b>early window</b> ({n_early} rows) vs <b>late window</b> ({n_late} rows)</p>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Method: Kolmogorov-Smirnov two-sample test (α = 0.05)</p>
    
    <div class="grid">
        <div class="card">
            <div class="metric">{len(drift_df)}</div>
            <div class="label">Features Analyzed</div>
        </div>
        <div class="card">
            <div class="metric" style="color: {'red' if n_drifted > 0 else 'green'};">{n_drifted}</div>
            <div class="label">Drift Detected</div>
        </div>
        <div class="card">
            <div class="metric" style="color: green;">{len(drift_df) - n_drifted}</div>
            <div class="label">No Drift</div>
        </div>
        <div class="card">
            <div class="metric">{n_drifted/len(drift_df)*100:.0f}%</div>
            <div class="label">Drift Rate</div>
        </div>
    </div>

    <h2>Feature-Level Drift Analysis</h2>
    <table>
        <tr>
            <th>Feature</th>
            <th>Early Mean</th>
            <th>Late Mean</th>
            <th>Early Std</th>
            <th>Late Std</th>
            <th>KS Statistic</th>
            <th>KS p-value</th>
            <th>Wasserstein Dist</th>
            <th>Status</th>
        </tr>
        {rows_html}
    </table>
    
    <h2>Methodology</h2>
    <div class="summary">
        <p><b>Kolmogorov-Smirnov Test:</b> Non-parametric test comparing the cumulative distributions 
        of the early and late windows. Drift is flagged when p-value &lt; 0.05.</p>
        <p><b>Wasserstein Distance:</b> Earth mover's distance measuring the minimum cost to transform 
        one distribution into another. Higher values indicate greater distributional shift.</p>
    </div>
</body>
</html>"""
    return html


def generate_quality_html(quality_df, total_rows):
    rows_html = ""
    for _, r in quality_df.iterrows():
        missing_color = "#ffcccc" if r["missing_pct"] > 1 else ""
        rows_html += f"""
        <tr>
            <td>{r['feature']}</td>
            <td>{r['count']}</td>
            <td style="background-color: {missing_color};">{r['missing']} ({r['missing_pct']:.1f}%)</td>
            <td>{r['zeros']} ({r['zeros_pct']:.1f}%)</td>
            <td>{r['mean']:.8f}</td>
            <td>{r['std']:.8f}</td>
            <td>{r['min']:.8f}</td>
            <td>{r['q25']:.8f}</td>
            <td>{r['q50']:.8f}</td>
            <td>{r['q75']:.8f}</td>
            <td>{r['max']:.8f}</td>
        </tr>"""
    
    total_missing = quality_df["missing"].sum()
    total_cells = quality_df["count"].sum()
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Data Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #fafafa; }}
        h1 {{ color: #1a1a2e; }}
        h2 {{ color: #16213e; margin-top: 30px; }}
        .summary {{ background: #fff; padding: 20px; border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th {{ background: #1a1a2e; color: white; padding: 10px 6px; text-align: left; font-size: 12px; }}
        td {{ padding: 7px 6px; border-bottom: 1px solid #ddd; font-size: 11px; }}
        .metric {{ font-size: 28px; font-weight: bold; color: #1a1a2e; }}
        .label {{ font-size: 14px; color: #666; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .card {{ background: #fff; padding: 20px; border-radius: 8px; text-align: center;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h1>📋 Data Quality Report</h1>
    <p>Dataset: features.parquet | {total_rows} rows | {len(quality_df)} features</p>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="grid">
        <div class="card">
            <div class="metric">{total_rows:,}</div>
            <div class="label">Total Rows</div>
        </div>
        <div class="card">
            <div class="metric">{len(quality_df)}</div>
            <div class="label">Features</div>
        </div>
        <div class="card">
            <div class="metric">{total_missing}</div>
            <div class="label">Missing Values</div>
        </div>
        <div class="card">
            <div class="metric" style="color: green;">{(1 - total_missing/total_cells)*100:.1f}%</div>
            <div class="label">Completeness</div>
        </div>
    </div>

    <h2>Feature Quality Summary</h2>
    <table>
        <tr>
            <th>Feature</th>
            <th>Count</th>
            <th>Missing</th>
            <th>Zeros</th>
            <th>Mean</th>
            <th>Std</th>
            <th>Min</th>
            <th>Q25</th>
            <th>Q50</th>
            <th>Q75</th>
            <th>Max</th>
        </tr>
        {rows_html}
    </table>
</body>
</html>"""
    return html


# Save reports
drift_html = generate_drift_html(drift_df, len(df_early), len(df_late))
drift_path = REPORTS_DIR / "data_drift_early_vs_late.html"
with open(drift_path, "w", encoding="utf-8") as f:
    f.write(drift_html)
print(f"[DONE] Drift report saved: {drift_path}")

quality_html = generate_quality_html(quality_df, len(df))
quality_path = REPORTS_DIR / "data_quality.html"
with open(quality_path, "w", encoding="utf-8") as f:
    f.write(quality_html)
print(f"[DONE] Quality report saved: {quality_path}")

# Also save as JSON for reference
drift_df.to_json(REPORTS_DIR / "data_drift.json", orient="records", indent=2)
quality_df.to_json(REPORTS_DIR / "data_quality.json", orient="records", indent=2)
print(f"[DONE] JSON reports also saved to {REPORTS_DIR}/")
