"""
PitWall — Offline Telemetry Extraction Script
==============================================
Extracts pit-stop deltas, tyre degradation curves, and Safety-Car
statistics from 4 F1 seasons (2022-2025) using the FastF1 library.

Output: data/pitwall_telemetry.db  (SQLite, ~4 MB)

This script is meant to be run ONCE offline.  It downloads session
data from the F1 timing servers (first run caches to disk; subsequent
runs are instant).

Usage:
    python scripts/extract_telemetry.py
"""

import sqlite3
import os
import sys
import warnings
from pathlib import Path

import time as _time

import fastf1
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "pitwall_telemetry.db"
CACHE_DIR = PROJECT_ROOT / "fastf1_cache"

# Enable FastF1 disk cache so re-runs are instant
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# ---------------------------------------------------------------------------
# Seasons & calendar
# ---------------------------------------------------------------------------
SEASONS = [2022, 2023, 2024, 2025]

# Pirelli compound naming: C1 (hardest) to C5 (softest)
# FastF1 uses simplified names: HARD, MEDIUM, SOFT
COMPOUND_MAP = {"HARD": "Hard", "MEDIUM": "Medium", "SOFT": "Soft"}


def create_database(conn: sqlite3.Connection):
    """Create the SQLite schema."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS circuits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            round_number INTEGER NOT NULL,
            circuit_name TEXT NOT NULL,
            event_name TEXT NOT NULL,
            country TEXT NOT NULL,
            total_laps INTEGER,
            UNIQUE(season, round_number)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pit_stop_deltas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            circuit_id INTEGER NOT NULL,
            driver TEXT NOT NULL,
            lap_number INTEGER NOT NULL,
            pit_duration_seconds REAL,
            compound_before TEXT,
            compound_after TEXT,
            FOREIGN KEY (circuit_id) REFERENCES circuits(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tyre_stints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            circuit_id INTEGER NOT NULL,
            driver TEXT NOT NULL,
            stint_number INTEGER NOT NULL,
            compound TEXT NOT NULL,
            start_lap INTEGER NOT NULL,
            end_lap INTEGER NOT NULL,
            stint_length INTEGER NOT NULL,
            avg_lap_time_seconds REAL,
            degradation_per_lap REAL,
            FOREIGN KEY (circuit_id) REFERENCES circuits(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS safety_cars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            circuit_id INTEGER NOT NULL,
            sc_type TEXT NOT NULL,
            start_lap INTEGER,
            end_lap INTEGER,
            FOREIGN KEY (circuit_id) REFERENCES circuits(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS circuit_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            circuit_name TEXT NOT NULL,
            avg_green_flag_pit_loss REAL,
            avg_pit_duration REAL,
            sc_probability REAL,
            total_races_sampled INTEGER,
            avg_soft_deg_per_lap REAL,
            avg_medium_deg_per_lap REAL,
            avg_hard_deg_per_lap REAL,
            UNIQUE(circuit_name)
        )
    """)

    conn.commit()
    print("[DB] Schema created successfully.")


def extract_race_data(season: int, conn: sqlite3.Connection):
    """Extract pit stops, tyre stints, and SC data for all races in a season."""
    cursor = conn.cursor()

    print(f"\n{'='*60}")
    print(f"  SEASON {season}")
    print(f"{'='*60}")

    for attempt in range(3):
        try:
            schedule = fastf1.get_event_schedule(season, include_testing=False)
            break
        except Exception as e:
            if "500 calls" in str(e) or "rate" in str(e).lower():
                wait_mins = 10 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait_mins} minutes before retrying...")
                _time.sleep(wait_mins * 60)
            else:
                print(f"  [ERROR] Could not load {season} schedule: {e}")
                return
    else:
        print(f"  [ERROR] Failed to load {season} schedule after 3 retries.")
        return

    # Filter to only conventional race rounds (round > 0)
    races = schedule[schedule["RoundNumber"] > 0]

    for _, event in races.iterrows():
        round_num = int(event["RoundNumber"])
        event_name = event["EventName"]
        country = event.get("Country", "Unknown")
        circuit_name = event.get("Location", event_name)

        print(f"\n  [{season} R{round_num:02d}] {event_name} ({circuit_name})...", end=" ")

        try:
            session = fastf1.get_session(season, round_num, "R")
            session.load(
                telemetry=False,  # Skip heavy telemetry to save RAM
                weather=False,
                messages=False,
            )
        except Exception as e:
            err_str = str(e)
            if "500 calls" in err_str or "rate" in err_str.lower():
                print(f"RATE LIMITED — waiting 10 minutes...")
                _time.sleep(600)
                try:
                    session = fastf1.get_session(season, round_num, "R")
                    session.load(telemetry=False, weather=False, messages=False)
                except Exception as e2:
                    print(f"SKIP after retry ({e2})")
                    continue
            else:
                print(f"SKIP ({e})")
                continue

        # --- Insert circuit record ---
        total_laps = int(session.total_laps) if hasattr(session, "total_laps") and session.total_laps else None
        cursor.execute(
            "INSERT OR IGNORE INTO circuits (season, round_number, circuit_name, event_name, country, total_laps) VALUES (?, ?, ?, ?, ?, ?)",
            (season, round_num, circuit_name, event_name, country, total_laps),
        )
        conn.commit()

        circuit_id = cursor.execute(
            "SELECT id FROM circuits WHERE season=? AND round_number=?",
            (season, round_num),
        ).fetchone()[0]

        # --- Laps data ---
        laps = session.laps
        if laps is None or laps.empty:
            print("NO LAPS DATA")
            continue

        # --- Pit stops ---
        # For each driver, find laps where PitInTime is set (in-lap) and
        # match with the next lap's PitOutTime (out-lap) for accurate
        # stationary pit duration.
        for driver in laps["Driver"].unique():
            d_laps = laps[laps["Driver"] == driver].sort_values("LapNumber")
            pit_in_laps = d_laps[d_laps["PitInTime"].notna()]

            for _, pit in pit_in_laps.iterrows():
                lap_num = int(pit["LapNumber"]) if pd.notna(pit["LapNumber"]) else 0
                compound_before = pit.get("Compound", None)

                # Find the out-lap (next lap for this driver)
                out_laps = d_laps[d_laps["LapNumber"] == lap_num + 1]

                pit_dur = None
                compound_after = None

                if not out_laps.empty:
                    out_lap = out_laps.iloc[0]
                    compound_after = out_lap.get("Compound", None)

                    # PitOutTime (out-lap) - PitInTime (in-lap) = stationary time
                    if pd.notna(out_lap.get("PitOutTime")) and pd.notna(pit.get("PitInTime")):
                        delta = out_lap["PitOutTime"] - pit["PitInTime"]
                        dur = delta.total_seconds() if hasattr(delta, "total_seconds") else None
                        # Sanity check: pit stops are typically 15-60 seconds
                        if dur is not None and 10.0 < dur < 120.0:
                            pit_dur = dur

                cursor.execute(
                    "INSERT INTO pit_stop_deltas (circuit_id, driver, lap_number, pit_duration_seconds, compound_before, compound_after) VALUES (?, ?, ?, ?, ?, ?)",
                    (circuit_id, driver, lap_num, pit_dur, compound_before, compound_after),
                )

        # --- Tyre stints ---
        for driver in laps["Driver"].unique():
            driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber").copy()
            if driver_laps.empty:
                continue

            # Identify stint boundaries by compound changes or pit stops
            driver_laps["StintChange"] = (
                (driver_laps["Compound"] != driver_laps["Compound"].shift(1))
                | (driver_laps["PitOutTime"].notna())
            )
            driver_laps["StintId"] = driver_laps["StintChange"].cumsum()

            for stint_id, stint_group in driver_laps.groupby("StintId"):
                compound = stint_group["Compound"].iloc[0]
                if pd.isna(compound):
                    continue

                start_lap = int(stint_group["LapNumber"].min())
                end_lap = int(stint_group["LapNumber"].max())
                stint_length = end_lap - start_lap + 1

                # Calculate lap times (exclude outliers: pit in/out laps, SC laps)
                valid_laps = stint_group[
                    stint_group["LapTime"].notna()
                    & ~stint_group["PitInTime"].notna()
                    & ~stint_group["PitOutTime"].notna()
                ].copy()

                if valid_laps.empty or stint_length < 3:
                    continue

                lap_times = valid_laps["LapTime"].dt.total_seconds()

                # Filter extreme outliers (SC laps, formation laps)
                median_time = lap_times.median()
                lap_times = lap_times[(lap_times > median_time * 0.95) & (lap_times < median_time * 1.10)]

                if len(lap_times) < 3:
                    continue

                avg_lap_time = float(lap_times.mean())

                # Degradation: linear regression slope of lap time vs lap number
                x = np.arange(len(lap_times))
                if len(x) >= 3:
                    coeffs = np.polyfit(x, lap_times.values, 1)
                    deg_per_lap = float(coeffs[0])  # seconds per lap of degradation
                else:
                    deg_per_lap = None

                cursor.execute(
                    "INSERT INTO tyre_stints (circuit_id, driver, stint_number, compound, start_lap, end_lap, stint_length, avg_lap_time_seconds, degradation_per_lap) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (circuit_id, driver, int(stint_id), compound, start_lap, end_lap, stint_length, avg_lap_time, deg_per_lap),
                )

        conn.commit()
        pit_count = cursor.execute(
            "SELECT COUNT(*) FROM pit_stop_deltas WHERE circuit_id=?", (circuit_id,)
        ).fetchone()[0]
        stint_count = cursor.execute(
            "SELECT COUNT(*) FROM tyre_stints WHERE circuit_id=?", (circuit_id,)
        ).fetchone()[0]
        print(f"OK ({pit_count} pits, {stint_count} stints)")


def build_circuit_summaries(conn: sqlite3.Connection):
    """Aggregate per-circuit summary statistics from raw data."""
    cursor = conn.cursor()

    print(f"\n{'='*60}")
    print(f"  BUILDING CIRCUIT SUMMARIES")
    print(f"{'='*60}")

    cursor.execute("DELETE FROM circuit_summaries")

    cursor.execute("""
        INSERT INTO circuit_summaries (
            circuit_name,
            avg_green_flag_pit_loss,
            avg_pit_duration,
            sc_probability,
            total_races_sampled,
            avg_soft_deg_per_lap,
            avg_medium_deg_per_lap,
            avg_hard_deg_per_lap
        )
        SELECT
            c.circuit_name,
            AVG(p.pit_duration_seconds) + 15.0 AS avg_green_flag_pit_loss,
            AVG(p.pit_duration_seconds) AS avg_pit_duration,
            0.0 AS sc_probability,
            COUNT(DISTINCT c.id) AS total_races_sampled,
            AVG(CASE WHEN t.compound = 'SOFT' THEN t.degradation_per_lap END) AS avg_soft_deg_per_lap,
            AVG(CASE WHEN t.compound = 'MEDIUM' THEN t.degradation_per_lap END) AS avg_medium_deg_per_lap,
            AVG(CASE WHEN t.compound = 'HARD' THEN t.degradation_per_lap END) AS avg_hard_deg_per_lap
        FROM circuits c
        LEFT JOIN pit_stop_deltas p ON p.circuit_id = c.id
        LEFT JOIN tyre_stints t ON t.circuit_id = c.id
        GROUP BY c.circuit_name
    """)

    conn.commit()

    # Print summary
    rows = cursor.execute(
        "SELECT circuit_name, total_races_sampled, avg_pit_duration, avg_soft_deg_per_lap, avg_medium_deg_per_lap, avg_hard_deg_per_lap FROM circuit_summaries ORDER BY circuit_name"
    ).fetchall()

    print(f"\n  {'Circuit':<30} {'Races':>6} {'Pit(s)':>8} {'Soft Deg':>10} {'Med Deg':>10} {'Hard Deg':>10}")
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for row in rows:
        name, races, pit, sdeg, mdeg, hdeg = row
        pit_s = f"{pit:.1f}" if pit else "N/A"
        s_s = f"{sdeg:.4f}" if sdeg else "N/A"
        m_s = f"{mdeg:.4f}" if mdeg else "N/A"
        h_s = f"{hdeg:.4f}" if hdeg else "N/A"
        print(f"  {name:<30} {races:>6} {pit_s:>8} {s_s:>10} {m_s:>10} {h_s:>10}")

    total_circuits = len(rows)
    total_races = sum(r[1] for r in rows)
    print(f"\n  TOTAL: {total_circuits} unique circuits, {total_races} race records across 4 seasons.")


def main():
    print("=" * 60)
    print("  PitWall Telemetry Extraction (2022-2025)")
    print("=" * 60)
    print(f"  Database: {DB_PATH}")
    print(f"  Cache:    {CACHE_DIR}")

    conn = sqlite3.connect(str(DB_PATH))
    create_database(conn)

    for season in SEASONS:
        extract_race_data(season, conn)

    build_circuit_summaries(conn)

    # Final stats
    cursor = conn.cursor()
    circuits = cursor.execute("SELECT COUNT(*) FROM circuits").fetchone()[0]
    pits = cursor.execute("SELECT COUNT(*) FROM pit_stop_deltas").fetchone()[0]
    stints = cursor.execute("SELECT COUNT(*) FROM tyre_stints").fetchone()[0]
    summaries = cursor.execute("SELECT COUNT(*) FROM circuit_summaries").fetchone()[0]

    print(f"\n{'='*60}")
    print(f"  EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Circuits:  {circuits}")
    print(f"  Pit Stops: {pits}")
    print(f"  Stints:    {stints}")
    print(f"  Summaries: {summaries}")
    print(f"  DB Size:   {DB_PATH.stat().st_size / 1024:.1f} KB")
    print(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()
