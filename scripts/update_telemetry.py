"""
PitWall — Manual Telemetry Updater
===================================
Fetches missing rounds for the current season (2026).
Queries the DB to find the latest round we have, and fetches anything new.

Usage:
    python scripts/update_telemetry.py
"""

import sqlite3
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_telemetry import extract_race_data, build_circuit_summaries, DB_PATH

def main():
    print("=" * 60)
    print("  PitWall Telemetry Updater (2026)")
    print("=" * 60)

    season = 2026
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Find the maximum round number we currently have for 2026
    try:
        max_round = cursor.execute(
            "SELECT MAX(round_number) FROM circuits WHERE season=?", (season,)
        ).fetchone()[0]
    except Exception:
        max_round = None

    if max_round is None:
        print(f"[INFO] No data found for {season}. Starting from Round 1.")
        start_round = 1
    else:
        # Start fetching from the next round
        start_round = max_round + 1
        print(f"[INFO] Found data up to Round {max_round}. Fetching from Round {start_round} onwards.")

    # Extract new rounds
    extract_race_data(season, conn, start_round=start_round)

    # Rebuild summaries to include new data
    build_circuit_summaries(conn)

    conn.close()
    print("=" * 60)
    print("  UPDATE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
