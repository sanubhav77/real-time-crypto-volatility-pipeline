"""
ws_ingest.py – Coinbase Advanced Trade WebSocket → Kafka producer

Connects to the Coinbase Advanced Trade WebSocket API using JWT auth,
subscribes to ticker data for one or more trading pairs, and publishes
each tick to the Kafka topic `ticks.raw`. Also mirrors raw messages to data/raw/.

Features:
  - JWT authentication for CDP API keys
  - Automatic reconnect with exponential backoff
  - Heartbeat monitoring
  - Graceful shutdown on Ctrl+C
  - Local NDJSON mirror of raw ticks

Usage:
  python scripts/ws_ingest.py --pair BTC-USD --minutes 15
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
import secrets
from datetime import datetime, timezone
from pathlib import Path

import jwt
import websockets
from cryptography.hazmat.primitives import serialization
from kafka import KafkaProducer
from dotenv import load_dotenv

# ── Configuration ───────────────────────────────────────────────────
load_dotenv()

COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC = "ticks.raw"
RAW_DATA_DIR = Path("data/raw")

# ── Globals ─────────────────────────────────────────────────────────
shutdown_event = asyncio.Event()


def load_private_key():
    """Load the EC private key from the environment variable."""
    raw = os.getenv("COINBASE_API_SECRET", "")
    # Handle escaped newlines from .env
    pem_str = raw.replace("\\n", "\n")
    private_key = serialization.load_pem_private_key(
        pem_str.encode("utf-8"), password=None
    )
    return private_key


def build_jwt_token() -> str:
    """Generate a JWT token for Coinbase Advanced Trade WebSocket."""
    api_key = os.getenv("COINBASE_API_KEY", "")
    private_key = load_private_key()

    now = int(time.time())
    payload = {
        "sub": api_key,
        "iss": "coinbase-cloud",
        "aud": ["cdp_service"],
        "nbf": now,
        "exp": now + 120,  # 2 minute expiry
        "nonce": secrets.token_hex(16),
    }

    headers = {
        "kid": api_key,
        "nonce": secrets.token_hex(16),
        "typ": "JWT",
    }

    token = jwt.encode(
        payload,
        private_key,
        algorithm="ES256",
        headers=headers,
    )
    return token


def build_producer() -> KafkaProducer:
    """Create and return a KafkaProducer with JSON serialization."""
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
    )


def build_subscribe_msg(pairs: list, channel: str = "ticker") -> dict:
    """Build the subscription message with JWT authentication."""
    token = build_jwt_token()
    msg = {
        "type": "subscribe",
        "product_ids": pairs,
        "channel": channel,
        "jwt": token,
    }
    return msg


def open_ndjson_writer(pairs: list):
    """Open a NDJSON file for mirroring raw ticks."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pair_tag = "_".join(p.replace("-", "") for p in pairs)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filepath = RAW_DATA_DIR / f"ticks_{pair_tag}_{ts}.ndjson"
    return open(filepath, "a", encoding="utf-8")


async def ingest(pairs: list, duration_seconds: int):
    """
    Main ingest loop.
    Connects to Coinbase WS, subscribes, and publishes ticks to Kafka.
    Implements reconnect with exponential backoff.
    """
    producer = build_producer()
    ndjson_file = open_ndjson_writer(pairs)
    msg_count = 0
    start_time = time.time()
    backoff = 1  # seconds, for reconnect

    print(f"[INFO] Starting ingest for {pairs}, duration={duration_seconds}s")
    print(f"[INFO] Kafka broker: {KAFKA_BOOTSTRAP}, topic: {KAFKA_TOPIC}")
    print(f"[INFO] Raw mirror: {ndjson_file.name}")

    while not shutdown_event.is_set():
        elapsed = time.time() - start_time
        if elapsed >= duration_seconds:
            print(f"\n[INFO] Duration reached ({duration_seconds}s). Stopping.")
            break

        try:
            async with websockets.connect(
                COINBASE_WS_URL, ping_interval=20, ping_timeout=10
            ) as ws:
                print(f"[INFO] Connected to {COINBASE_WS_URL}")
                backoff = 1  # reset on successful connection

                # Subscribe to ticker channel
                sub_msg = build_subscribe_msg(pairs, "ticker")
                await ws.send(json.dumps(sub_msg))
                print(f"[INFO] Subscribed to ticker for {pairs}")

                # Also subscribe to heartbeats
                hb_msg = build_subscribe_msg(pairs, "heartbeats")
                await ws.send(json.dumps(hb_msg))
                print(f"[INFO] Subscribed to heartbeats")

                last_heartbeat = time.time()

                while not shutdown_event.is_set():
                    # Check duration
                    elapsed = time.time() - start_time
                    if elapsed >= duration_seconds:
                        break

                    # Check heartbeat timeout (90 seconds)
                    if time.time() - last_heartbeat > 90:
                        print("[WARN] No heartbeat for 90s, reconnecting...")
                        break

                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        print("[WARN] No message for 30s, still waiting...")
                        continue

                    data = json.loads(raw)
                    channel = data.get("channel", "")

                    # Update heartbeat timer
                    if channel == "heartbeats":
                        last_heartbeat = time.time()
                        continue

                    # Skip subscription confirmations
                    if channel == "subscriptions":
                        print(f"[INFO] Subscription confirmed: {data}")
                        continue

                    # Process ticker events
                    if channel == "ticker":
                        events = data.get("events", [])
                        for event in events:
                            tickers = event.get("tickers", [])
                            for tick in tickers:
                                enriched = {
                                    "timestamp": data.get("timestamp", ""),
                                    "sequence_num": data.get(
                                        "sequence_num", 0
                                    ),
                                    "product_id": tick.get("product_id", ""),
                                    "price": tick.get("price", ""),
                                    "best_bid": tick.get("best_bid", ""),
                                    "best_ask": tick.get("best_ask", ""),
                                    "volume_24_h": tick.get(
                                        "volume_24_h", ""
                                    ),
                                    "low_24_h": tick.get("low_24_h", ""),
                                    "high_24_h": tick.get("high_24_h", ""),
                                    "price_percent_chg_24_h": tick.get(
                                        "price_percent_chg_24_h", ""
                                    ),
                                    "best_bid_quantity": tick.get(
                                        "best_bid_quantity", ""
                                    ),
                                    "best_ask_quantity": tick.get(
                                        "best_ask_quantity", ""
                                    ),
                                    "type": event.get("type", ""),
                                    "ingest_ts": datetime.now(
                                        timezone.utc
                                    ).isoformat(),
                                }

                                # Publish to Kafka
                                producer.send(KAFKA_TOPIC, value=enriched)
                                msg_count += 1

                                # Mirror to NDJSON
                                ndjson_file.write(
                                    json.dumps(enriched) + "\n"
                                )

                                if msg_count % 50 == 0:
                                    ndjson_file.flush()
                                    producer.flush()
                                    elapsed = time.time() - start_time
                                    print(
                                        f"[INFO] {msg_count} ticks ingested "
                                        f"| elapsed: {elapsed:.0f}s / {duration_seconds}s"
                                    )

        except websockets.exceptions.ConnectionClosedError as e:
            print(
                f"[WARN] Connection closed: {e}. Reconnecting in {backoff}s..."
            )
        except Exception as e:
            print(
                f"[ERROR] {type(e).__name__}: {e}. Reconnecting in {backoff}s..."
            )

        if not shutdown_event.is_set() and (
            time.time() - start_time
        ) < duration_seconds:
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)  # exponential backoff, max 30s

    # ── Cleanup ─────────────────────────────────────────────────────
    producer.flush()
    producer.close()
    ndjson_file.close()
    print(f"\n[DONE] Total ticks ingested: {msg_count}")
    print(f"[DONE] Raw data saved to: {ndjson_file.name}")


def handle_shutdown(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n[INFO] Shutdown signal received...")
    shutdown_event.set()


def main():
    parser = argparse.ArgumentParser(
        description="Coinbase WS → Kafka ingestor"
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="BTC-USD",
        help="Trading pair(s), comma-separated (e.g., BTC-USD,ETH-USD)",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=15,
        help="How many minutes to run the ingestor (default: 15)",
    )
    args = parser.parse_args()

    pairs = [p.strip() for p in args.pair.split(",")]
    duration_seconds = args.minutes * 60

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    asyncio.run(ingest(pairs, duration_seconds))


if __name__ == "__main__":
    main()
