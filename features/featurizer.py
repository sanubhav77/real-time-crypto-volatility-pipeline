"""
featurizer.py – Kafka consumer that computes windowed features

Reads from ticks.raw, computes rolling features (midprice returns,
bid-ask spread, trade intensity, volatility), and outputs to
ticks.features topic + Parquet file.

Can run in two modes:
  1. Live: reads from Kafka topic ticks.raw
  2. Replay: reads from NDJSON file (called by replay.py)

Usage:
  python features/featurizer.py --topic_in ticks.raw --topic_out ticks.features
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from dotenv import load_dotenv

load_dotenv()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
PROCESSED_DIR = Path("data/processed")


class FeatureEngine:
    """Computes rolling windowed features from tick data."""

    def __init__(self, windows=(30, 60, 120)):
        self.windows = windows
        self.max_window = max(windows)
        # Store recent ticks as (timestamp_epoch, midprice, bid, ask, price)
        self.tick_buffer = deque()
        self.midprice_returns = deque()
        self.prev_midprice = None
        self.tick_count_buffer = deque()  # (timestamp_epoch,) for trade intensity

    def _parse_timestamp(self, ts_str: str) -> float:
        """Parse ISO timestamp to epoch seconds."""
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, AttributeError):
            return time.time()

    def _clean_buffer(self, current_time: float):
        """Remove ticks older than max_window from buffers."""
        cutoff = current_time - self.max_window - 10  # small buffer
        while self.tick_buffer and self.tick_buffer[0][0] < cutoff:
            self.tick_buffer.popleft()
        while self.midprice_returns and self.midprice_returns[0][0] < cutoff:
            self.midprice_returns.popleft()
        while self.tick_count_buffer and self.tick_count_buffer[0] < cutoff:
            self.tick_count_buffer.popleft()

    def compute_features(self, tick: dict) -> dict | None:
        """
        Compute features for a single tick.
        Returns a feature dict, or None if not enough data yet.
        """
        ts_str = tick.get("timestamp", "")
        ts_epoch = self._parse_timestamp(ts_str)

        # Parse numeric fields
        try:
            best_bid = float(tick.get("best_bid", 0))
            best_ask = float(tick.get("best_ask", 0))
            price = float(tick.get("price", 0))
        except (ValueError, TypeError):
            return None

        if best_bid <= 0 or best_ask <= 0:
            return None

        midprice = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        spread_bps = (spread / midprice) * 10000 if midprice > 0 else 0

        # Compute midprice return
        mid_return = 0.0
        if self.prev_midprice is not None and self.prev_midprice > 0:
            mid_return = (midprice - self.prev_midprice) / self.prev_midprice
        self.prev_midprice = midprice

        # Store in buffers
        self.tick_buffer.append((ts_epoch, midprice, best_bid, best_ask, price))
        self.midprice_returns.append((ts_epoch, mid_return))
        self.tick_count_buffer.append(ts_epoch)

        # Clean old data
        self._clean_buffer(ts_epoch)

        # Need at least some data to compute rolling features
        if len(self.midprice_returns) < 5:
            return None

        # Compute rolling features for each window
        features = {
            "timestamp": ts_str,
            "timestamp_epoch": ts_epoch,
            "product_id": tick.get("product_id", ""),
            "price": price,
            "midprice": midprice,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_bps": round(spread_bps, 4),
            "mid_return": round(mid_return, 8),
        }

        for w in self.windows:
            cutoff = ts_epoch - w
            # Filter returns within window
            window_returns = [r for t, r in self.midprice_returns if t >= cutoff]
            window_ticks = [(t, m) for t, m, b, a, p in self.tick_buffer if t >= cutoff]

            if len(window_returns) >= 2:
                returns_arr = np.array(window_returns)
                features[f"volatility_{w}s"] = round(float(np.std(returns_arr)), 8)
                features[f"mean_return_{w}s"] = round(float(np.mean(returns_arr)), 8)
                features[f"max_return_{w}s"] = round(float(np.max(np.abs(returns_arr))), 8)
            else:
                features[f"volatility_{w}s"] = 0.0
                features[f"mean_return_{w}s"] = 0.0
                features[f"max_return_{w}s"] = 0.0

            # Trade intensity: ticks per second in window
            ticks_in_window = sum(1 for t in self.tick_count_buffer if t >= cutoff)
            features[f"tick_intensity_{w}s"] = round(ticks_in_window / w, 4) if w > 0 else 0

            # Spread stats in window
            window_spreads = [a - b for t, m, b, a, p in self.tick_buffer if t >= cutoff and a > b]
            if window_spreads:
                features[f"mean_spread_{w}s"] = round(float(np.mean(window_spreads)), 4)
            else:
                features[f"mean_spread_{w}s"] = 0.0

        # Order book imbalance (current tick)
        bid_qty = float(tick.get("best_bid_quantity", 0) or 0)
        ask_qty = float(tick.get("best_ask_quantity", 0) or 0)
        total_qty = bid_qty + ask_qty
        features["book_imbalance"] = round((bid_qty - ask_qty) / total_qty, 6) if total_qty > 0 else 0.0

        return features


def compute_labels(df: pd.DataFrame, horizon_seconds: float = 60.0) -> pd.DataFrame:
    """
    Compute forward-looking volatility labels.
    Label = 1 if rolling std of midprice returns in the NEXT horizon >= threshold.
    Threshold is set later during EDA.
    """
    df = df.copy()
    df = df.sort_values("timestamp_epoch").reset_index(drop=True)

    # Compute future volatility for each row
    future_vols = []
    ts_arr = df["timestamp_epoch"].values
    ret_arr = df["mid_return"].values

    for i in range(len(df)):
        current_ts = ts_arr[i]
        future_cutoff = current_ts + horizon_seconds
        # Get returns in the future window
        mask = (ts_arr > current_ts) & (ts_arr <= future_cutoff)
        future_returns = ret_arr[mask]
        if len(future_returns) >= 2:
            future_vols.append(float(np.std(future_returns)))
        else:
            future_vols.append(np.nan)

    df["future_volatility_60s"] = future_vols
    return df


def run_live(topic_in: str, topic_out: str, max_messages: int = None):
    """Run featurizer in live Kafka consumer mode."""
    consumer = KafkaConsumer(
        topic_in,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="featurizer-group",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=10000,
    )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    engine = FeatureEngine()
    all_features = []
    count = 0

    print(f"[INFO] Featurizer reading from '{topic_in}' → '{topic_out}'")

    for msg in consumer:
        tick = msg.value
        features = engine.compute_features(tick)
        if features:
            producer.send(topic_out, value=features)
            all_features.append(features)
            count += 1
            if count % 200 == 0:
                print(f"[INFO] {count} feature rows computed")

        if max_messages and count >= max_messages:
            break

    producer.flush()
    producer.close()
    consumer.close()

    # Save to Parquet
    if all_features:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_features)
        df = compute_labels(df)
        out_path = PROCESSED_DIR / "features.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[DONE] {count} feature rows saved to {out_path}")
        print(f"[DONE] Columns: {list(df.columns)}")
    else:
        print("[WARN] No features computed")


def run_from_dataframe(ticks: list[dict]) -> pd.DataFrame:
    """
    Run featurizer on a list of tick dicts (used by replay.py).
    Returns a DataFrame of features.
    """
    engine = FeatureEngine()
    all_features = []

    for tick in ticks:
        features = engine.compute_features(tick)
        if features:
            all_features.append(features)

    if all_features:
        df = pd.DataFrame(all_features)
        df = compute_labels(df)
        return df
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Featurizer: ticks → features")
    parser.add_argument("--topic_in", default="ticks.raw", help="Input Kafka topic")
    parser.add_argument("--topic_out", default="ticks.features", help="Output Kafka topic")
    parser.add_argument("--max_messages", type=int, default=None, help="Max messages to process")
    args = parser.parse_args()

    run_live(args.topic_in, args.topic_out, args.max_messages)


if __name__ == "__main__":
    main()
