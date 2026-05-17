"""
replay.py – Replay raw NDJSON data through the featurizer

Takes saved raw tick data and regenerates features identically
to the live Kafka consumer pipeline. Used to verify feature consistency.

Usage:
  python scripts/replay.py --raw data/raw/*.ndjson --out data/processed/features.parquet
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

# Add parent dir to path so we can import featurizer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.featurizer import run_from_dataframe


def load_ndjson_files(patterns: list[str]) -> list[dict]:
    """Load ticks from one or more NDJSON files."""
    all_ticks = []
    files_found = []

    for pattern in patterns:
        matched = glob.glob(pattern)
        files_found.extend(matched)

    if not files_found:
        print(f"[ERROR] No files found matching: {patterns}")
        sys.exit(1)

    for filepath in sorted(files_found):
        print(f"[INFO] Loading: {filepath}")
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        tick = json.loads(line)
                        all_ticks.append(tick)
                        count += 1
                    except json.JSONDecodeError:
                        continue
        print(f"[INFO]   → {count} ticks loaded")

    # Sort by timestamp for consistent ordering
    all_ticks.sort(key=lambda t: t.get("timestamp", ""))
    print(f"[INFO] Total ticks loaded: {len(all_ticks)}")
    return all_ticks


def main():
    parser = argparse.ArgumentParser(description="Replay raw ticks → features")
    parser.add_argument(
        "--raw",
        nargs="+",
        required=True,
        help="Path(s) or glob pattern(s) for raw NDJSON files",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed/features.parquet",
        help="Output Parquet file path",
    )
    args = parser.parse_args()

    # Load raw ticks
    ticks = load_ndjson_files(args.raw)

    if not ticks:
        print("[ERROR] No ticks to process")
        sys.exit(1)

    # Run through featurizer
    print("[INFO] Computing features via replay...")
    df = run_from_dataframe(ticks)

    if df.empty:
        print("[ERROR] No features generated")
        sys.exit(1)

    # Save to Parquet
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"\n[DONE] Replay complete!")
    print(f"[DONE] Features saved to: {out_path}")
    print(f"[DONE] Shape: {df.shape}")
    print(f"[DONE] Columns: {list(df.columns)}")

    # Quick stats
    print(f"\n[STATS] Time range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print(f"[STATS] Price range: {df['price'].min():.2f} → {df['price'].max():.2f}")
    vol_col = "future_volatility_60s"
    if vol_col in df.columns:
        valid = df[vol_col].dropna()
        print(f"[STATS] Future volatility (60s): mean={valid.mean():.8f}, "
              f"std={valid.std():.8f}, max={valid.max():.8f}")
        print(f"[STATS] Non-null labels: {len(valid)} / {len(df)}")


if __name__ == "__main__":
    main()
