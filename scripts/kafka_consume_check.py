"""
kafka_consume_check.py – Verify messages in a Kafka topic

Reads messages from a Kafka topic and prints summary statistics.
Used to validate that the ingestor is publishing data correctly.

Usage:
  python scripts/kafka_consume_check.py --topic ticks.raw --min 100
"""

import argparse
import json
import sys
from datetime import datetime

from kafka import KafkaConsumer, TopicPartition


def main():
    parser = argparse.ArgumentParser(description="Kafka topic consumer check")
    parser.add_argument(
        "--topic",
        type=str,
        default="ticks.raw",
        help="Kafka topic to consume from (default: ticks.raw)",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=100,
        help="Minimum expected message count (default: 100)",
    )
    parser.add_argument(
        "--bootstrap",
        type=str,
        default="localhost:9092",
        help="Kafka bootstrap server (default: localhost:9092)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=3,
        help="Number of sample messages to display (default: 3)",
    )
    args = parser.parse_args()

    print(f"[INFO] Connecting to Kafka at {args.bootstrap}")
    print(f"[INFO] Checking topic: {args.topic}")
    print(f"[INFO] Minimum expected messages: {args.min}")
    print("-" * 60)

    # Create consumer - read from the beginning
    consumer = KafkaConsumer(
        bootstrap_servers=args.bootstrap,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=5000,  # stop after 5s of no new messages
    )

    # Check if topic exists
    topics = consumer.topics()
    if args.topic not in topics:
        print(f"[FAIL] Topic '{args.topic}' not found!")
        print(f"[INFO] Available topics: {sorted(topics)}")
        consumer.close()
        sys.exit(1)

    # Assign to all partitions from beginning
    partitions = consumer.partitions_for_topic(args.topic)
    tp_list = [TopicPartition(args.topic, p) for p in partitions]
    consumer.assign(tp_list)
    consumer.seek_to_beginning()

    # Consume messages
    messages = []
    product_ids = set()
    timestamps = []

    print(f"[INFO] Reading messages from topic '{args.topic}'...")

    for msg in consumer:
        data = msg.value
        messages.append(data)
        product_ids.add(data.get("product_id", "unknown"))
        ts = data.get("timestamp", "")
        if ts:
            timestamps.append(ts)

    consumer.close()

    # ── Report ──────────────────────────────────────────────────────
    total = len(messages)
    print(f"\n{'=' * 60}")
    print(f"  KAFKA TOPIC CHECK REPORT")
    print(f"{'=' * 60}")
    print(f"  Topic:            {args.topic}")
    print(f"  Total messages:   {total}")
    print(f"  Product IDs:      {', '.join(sorted(product_ids))}")

    if timestamps:
        print(f"  First timestamp:  {min(timestamps)}")
        print(f"  Last timestamp:   {max(timestamps)}")

    print(f"  Minimum expected: {args.min}")

    if total >= args.min:
        print(f"\n  ✓ PASS — {total} messages >= {args.min} minimum")
    else:
        print(f"\n  ✗ FAIL — {total} messages < {args.min} minimum")

    # Show sample messages
    if messages and args.sample > 0:
        print(f"\n{'─' * 60}")
        print(f"  SAMPLE MESSAGES (first {min(args.sample, total)}):")
        print(f"{'─' * 60}")
        for i, msg in enumerate(messages[: args.sample]):
            print(f"\n  Message {i + 1}:")
            print(f"    product_id:  {msg.get('product_id')}")
            print(f"    price:       {msg.get('price')}")
            print(f"    best_bid:    {msg.get('best_bid')}")
            print(f"    best_ask:    {msg.get('best_ask')}")
            print(f"    timestamp:   {msg.get('timestamp')}")
            print(f"    type:        {msg.get('type')}")

    print(f"\n{'=' * 60}")

    sys.exit(0 if total >= args.min else 1)


if __name__ == "__main__":
    main()
