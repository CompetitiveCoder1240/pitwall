"""
PitWall — LangChain Tool Definitions
=====================================
Defines @tool functions that the LangGraph Agent can call:

1. query_telemetry  — Fast SQLite lookup for pit stop & tyre degradation data
2. calculate_strategy — Deterministic Python calculator for race stint projections
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from backend.logger import logger

# ---------------------------------------------------------------------------
# Database path — resolved relative to project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "pitwall_telemetry.db"
SUPABASE_URL = os.getenv("SUPABASE_URL")


def _query_db(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a read-only SQL query and return results as list of dicts.
    Uses Supabase (PostgreSQL) if SUPABASE_URL is set, otherwise falls back to local SQLite.
    """
    try:
        logger.debug(f"SQL Query: {sql[:200]}")
        
        if SUPABASE_URL:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            # Convert SQLite '?' placeholders to PostgreSQL '%s'
            pg_sql = sql.replace("?", "%s")
            conn = psycopg2.connect(SUPABASE_URL)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(pg_sql, params)
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return results
        else:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql, params)
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return results
    except Exception as e:
        logger.error(f"Database query failed: {e}", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Tool 1: Telemetry Database Query
# ---------------------------------------------------------------------------
@tool
def query_telemetry(
    circuit_name: str,
    query_type: str = "summary",
    compound: Optional[str] = None,
    season: Optional[int] = None,
) -> str:
    """Query the PitWall telemetry database for F1 circuit data from 2022-2025.

    Args:
        circuit_name: The circuit/location name (e.g. 'Monza', 'Silverstone', 'Sakhir', 'Monaco').
                      Use partial matching - the tool will search for circuits containing this name.
        query_type: One of:
            - 'summary': Average pit stop duration, tyre degradation rates, and race count for the circuit.
            - 'pit_stops': Detailed pit stop records (duration, compounds before/after) for the circuit.
            - 'tyre_stints': Detailed tyre stint data (degradation per lap, avg lap time) for the circuit.
            - 'all_circuits': List all available circuits and their race counts (ignores circuit_name).
        compound: Optional tyre compound filter for tyre_stints query ('SOFT', 'MEDIUM', 'HARD').
        season: Optional season year filter (2022, 2023, 2024, or 2025).

    Returns:
        Formatted string with the query results.
    """
    logger.info(f"Telemetry query: type={query_type}, circuit={circuit_name}")

    if query_type == "all_circuits":
        rows = _query_db(
            "SELECT circuit_name, total_races_sampled, avg_pit_duration, "
            "avg_soft_deg_per_lap, avg_medium_deg_per_lap, avg_hard_deg_per_lap "
            "FROM circuit_summaries ORDER BY circuit_name"
        )
        if not rows:
            logger.warning(f"No {query_type} data found for circuit: {circuit_name}")
            return "No circuit data found in the telemetry database."
        lines = ["Available circuits (2022-2025):\n"]
        for r in rows:
            lines.append(
                f"  {r['circuit_name']}: {r['total_races_sampled']} races, "
                f"avg pit duration: {r['avg_pit_duration']:.1f}s"
            )
        return "\n".join(lines)

    if query_type == "summary":
        rows = _query_db(
            "SELECT * FROM circuit_summaries WHERE circuit_name LIKE ?",
            (f"%{circuit_name}%",),
        )
        if not rows:
            logger.warning(f"No {query_type} data found for circuit: {circuit_name}")
            return f"No summary data found for circuit matching '{circuit_name}'."
        r = rows[0]
        result = (
            f"Circuit Summary: {r['circuit_name']}\n"
            f"  Total races sampled: {r['total_races_sampled']}\n"
            f"  Avg pit stop duration: {r['avg_pit_duration']:.1f}s\n"
            f"  Avg green-flag pit loss (pit duration + 15s in/out): {r['avg_green_flag_pit_loss']:.1f}s\n"
            f"  Avg Safety Car pit loss (green-flag loss - 12s SC advantage): {max(r['avg_green_flag_pit_loss'] - 12.0, 8.0):.1f}s\n"
        )
        if r["avg_soft_deg_per_lap"] is not None:
            result += f"  Avg Soft degradation: {r['avg_soft_deg_per_lap']:+.4f} sec/lap\n"
        if r["avg_medium_deg_per_lap"] is not None:
            result += f"  Avg Medium degradation: {r['avg_medium_deg_per_lap']:+.4f} sec/lap\n"
        if r["avg_hard_deg_per_lap"] is not None:
            result += f"  Avg Hard degradation: {r['avg_hard_deg_per_lap']:+.4f} sec/lap\n"
        return result

    if query_type == "pit_stops":
        sql = (
            "SELECT p.driver, p.lap_number, p.pit_duration_seconds, "
            "p.compound_before, p.compound_after, c.season, c.event_name "
            "FROM pit_stop_deltas p JOIN circuits c ON p.circuit_id = c.id "
            "WHERE c.circuit_name LIKE ?"
        )
        params = [f"%{circuit_name}%"]
        if season:
            sql += " AND c.season = ?"
            params.append(season)
        sql += " ORDER BY c.season, p.lap_number LIMIT 50"
        rows = _query_db(sql, tuple(params))
        if not rows:
            logger.warning(f"No {query_type} data found for circuit: {circuit_name}")
            return f"No pit stop data found for circuit matching '{circuit_name}'."
        lines = [f"Pit stops at {circuit_name} (showing up to 50):\n"]
        for r in rows:
            dur = f"{r['pit_duration_seconds']:.1f}s" if r["pit_duration_seconds"] else "N/A"
            lines.append(
                f"  {r['season']} {r['event_name']} | Driver {r['driver']} | "
                f"Lap {r['lap_number']} | Duration: {dur} | "
                f"{r['compound_before'] or '?'} → {r['compound_after'] or '?'}"
            )
        return "\n".join(lines)

    if query_type == "tyre_stints":
        sql = (
            "SELECT t.driver, t.compound, t.start_lap, t.end_lap, t.stint_length, "
            "t.avg_lap_time_seconds, t.degradation_per_lap, c.season, c.event_name "
            "FROM tyre_stints t JOIN circuits c ON t.circuit_id = c.id "
            "WHERE c.circuit_name LIKE ?"
        )
        params = [f"%{circuit_name}%"]
        if compound:
            sql += " AND t.compound = ?"
            params.append(compound.upper())
        if season:
            sql += " AND c.season = ?"
            params.append(season)
        sql += " ORDER BY c.season, t.driver, t.stint_number LIMIT 50"
        rows = _query_db(sql, tuple(params))
        if not rows:
            logger.warning(f"No {query_type} data found for circuit: {circuit_name}")
            return f"No tyre stint data found for circuit matching '{circuit_name}'."
        lines = [f"Tyre stints at {circuit_name} (showing up to 50):\n"]
        for r in rows:
            deg = f"{r['degradation_per_lap']:+.4f}s/lap" if r["degradation_per_lap"] else "N/A"
            avg = f"{r['avg_lap_time_seconds']:.2f}s" if r["avg_lap_time_seconds"] else "N/A"
            lines.append(
                f"  {r['season']} | Driver {r['driver']} | {r['compound']} | "
                f"Laps {r['start_lap']}-{r['end_lap']} ({r['stint_length']} laps) | "
                f"Avg: {avg} | Deg: {deg}"
            )
        return "\n".join(lines)

    logger.warning(f"Unknown query_type: {query_type}")
    return f"Unknown query_type '{query_type}'. Use: summary, pit_stops, tyre_stints, or all_circuits."


# ---------------------------------------------------------------------------
# Strategy Simulation Engine
# ---------------------------------------------------------------------------
# Simulates 1-stop and 2-stop race strategies using the Hybrid ML model.
# Enforces FIA mandatory rules: at least 1 pit stop, 2 different compounds.
#
# Two modes:
#   PRE-RACE  (current_lap <= 1):  Full simulation of all viable strategies.
#   LIVE      (current_lap > 1):   Pit window + compound recommendation.
# ---------------------------------------------------------------------------

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
MIN_STINT_LAPS = 8          # Minimum realistic F1 stint length
PIT_SWEEP_STEP = 2          # Sweep resolution for 1-stop (every 2 laps)
PIT_SWEEP_STEP_2 = 3        # Coarser sweep for 2-stop (O(n^2) so wider step)
CLIFF_DEG_THRESHOLD = 0.30  # s/lap marginal degradation = tyre cliff

# Realistic F1 strategy templates (FIA: must use >= 2 different compounds)
ONE_STOP_COMBOS = [
    ("SOFT", "HARD"), ("SOFT", "MEDIUM"),
    ("MEDIUM", "HARD"), ("MEDIUM", "SOFT"),
    ("HARD", "MEDIUM"), ("HARD", "SOFT"),
]

TWO_STOP_COMBOS = [
    ("SOFT", "MEDIUM", "HARD"),
    ("SOFT", "HARD", "SOFT"),
    ("SOFT", "MEDIUM", "SOFT"),
    ("MEDIUM", "HARD", "SOFT"),
    ("MEDIUM", "SOFT", "MEDIUM"),
    ("MEDIUM", "SOFT", "HARD"),
]


def _load_hybrid_models():
    """Load the Hybrid ML models for strategy simulation.

    Returns:
        dict with model objects and feature metadata, or None on failure.
    """
    try:
        import joblib
        from xgboost import XGBRegressor

        MODELS_DIR = PROJECT_ROOT / "models"
        mvr_path = MODELS_DIR / "mvr_anchor.pkl"
        xgb_path = MODELS_DIR / "xgb_booster.json"

        if mvr_path.exists() and xgb_path.exists():
            mvr_data = joblib.load(mvr_path)
            xgb_model = XGBRegressor()
            xgb_model.load_model(str(xgb_path))
            return {
                "mvr_model": mvr_data["mvr_model"],
                "mvr_features": mvr_data["mvr_features"],
                "circuit_cats": mvr_data["circuit_categories"],
                "compound_cats": mvr_data["compound_categories"],
                "xgb_model": xgb_model,
            }
    except Exception as e:
        logger.warning(f"Could not load hybrid models: {e}")
    return None


def _project_stint(circuit_name, compound, num_laps, race_lap_start,
                   total_race_laps, track_temp, driver, models,
                   tyre_life_start=1):
    """Project lap times for a single tyre stint using the Hybrid ML model.

    Args:
        circuit_name: Circuit short name (e.g. 'Monza').
        compound: 'SOFT', 'MEDIUM', or 'HARD'.
        num_laps: Stint length in laps.
        race_lap_start: Absolute race lap where this stint begins (1-indexed).
        total_race_laps: Total laps in the race (for fuel calculation).
        track_temp: Track surface temperature in Celsius.
        driver: 3-letter driver code.
        models: Dict from _load_hybrid_models().
        tyre_life_start: Starting tyre age (1=fresh, higher=used tyres).

    Returns:
        List of predicted lap times for the stint.
    """
    import pandas as pd
    import numpy as np

    if num_laps <= 0:
        return []

    mvr_model = models["mvr_model"]
    mvr_features = models["mvr_features"]
    circuit_cats = models["circuit_cats"]
    compound_cats = models["compound_cats"]
    xgb_model = models["xgb_model"]

    # Build the feature matrix for all laps in the stint
    laps_data = []
    for i in range(num_laps):
        race_lap = race_lap_start + i
        laps_data.append({
            "Circuit": circuit_name,
            "Compound": compound,
            "TyreLife": tyre_life_start + i,
            "FuelLoad": max(0, total_race_laps - race_lap),
            "TrackTemp": track_temp,
        })

    df_laps = pd.DataFrame(laps_data)

    # 1. Physics Anchor (MVR Linear Regression)
    df_mvr = pd.DataFrame(0, index=np.arange(num_laps), columns=mvr_features)
    df_mvr["FuelLoad"] = df_laps["FuelLoad"]
    df_mvr["TrackTemp"] = df_laps["TrackTemp"]

    circuit_col = f"Circuit_{circuit_name}"
    if circuit_col in df_mvr.columns:
        df_mvr[circuit_col] = 1
    compound_col = f"Compound_{compound}"
    if compound_col in df_mvr.columns:
        df_mvr[compound_col] = 1
    driver_col = f"Driver_{driver.upper()}"
    if driver_col in df_mvr.columns:
        df_mvr[driver_col] = 1

    linear_preds = mvr_model.predict(df_mvr)

    # 2. ML Booster (XGBoost Residual)
    df_xgb = df_laps[["Circuit", "Compound", "TyreLife", "TrackTemp"]].copy()
    df_xgb["Circuit"] = pd.Categorical(df_xgb["Circuit"], categories=circuit_cats)
    df_xgb["Compound"] = pd.Categorical(df_xgb["Compound"], categories=compound_cats)

    residual_preds = xgb_model.predict(df_xgb)

    return (linear_preds + residual_preds).tolist()


def _detect_cliff_lap(lap_times, threshold=CLIFF_DEG_THRESHOLD):
    """Detect the lap offset where tyre degradation exceeds threshold.

    Returns the index into lap_times where the cliff starts, or None.
    """
    for i in range(1, len(lap_times)):
        marginal_deg = lap_times[i] - lap_times[i - 1]
        if marginal_deg > threshold:
            return i
    return None


def _simulate_full_race(circuit_name, total_laps, pit_loss, track_temp,
                        driver, models):
    """Simulate all 1-stop and 2-stop strategies. Return sorted results."""
    results = []

    # ── 1-Stop Strategies ──
    for combo in ONE_STOP_COMBOS:
        c1, c2 = combo
        for pit_lap in range(MIN_STINT_LAPS,
                             total_laps - MIN_STINT_LAPS + 1,
                             PIT_SWEEP_STEP):
            stint1_len = pit_lap
            stint2_len = total_laps - pit_lap

            times1 = _project_stint(circuit_name, c1, stint1_len, 1,
                                    total_laps, track_temp, driver, models)
            times2 = _project_stint(circuit_name, c2, stint2_len, pit_lap + 1,
                                    total_laps, track_temp, driver, models)

            total_time = sum(times1) + pit_loss + sum(times2)

            results.append({
                "stops": 1,
                "compounds": [c1, c2],
                "pit_laps": [pit_lap],
                "stint_lengths": [stint1_len, stint2_len],
                "stint_avgs": [
                    sum(times1) / len(times1),
                    sum(times2) / len(times2),
                ],
                "stint_last": [times1[-1], times2[-1]],
                "total_time": total_time,
            })

    # ── 2-Stop Strategies ──
    for combo in TWO_STOP_COMBOS:
        c1, c2, c3 = combo
        for pit1 in range(MIN_STINT_LAPS,
                          total_laps - 2 * MIN_STINT_LAPS + 1,
                          PIT_SWEEP_STEP_2):
            for pit2 in range(pit1 + MIN_STINT_LAPS,
                              total_laps - MIN_STINT_LAPS + 1,
                              PIT_SWEEP_STEP_2):
                s1 = pit1
                s2 = pit2 - pit1
                s3 = total_laps - pit2

                t1 = _project_stint(circuit_name, c1, s1, 1,
                                    total_laps, track_temp, driver, models)
                t2 = _project_stint(circuit_name, c2, s2, pit1 + 1,
                                    total_laps, track_temp, driver, models)
                t3 = _project_stint(circuit_name, c3, s3, pit2 + 1,
                                    total_laps, track_temp, driver, models)

                total_time = sum(t1) + pit_loss + sum(t2) + pit_loss + sum(t3)

                results.append({
                    "stops": 2,
                    "compounds": [c1, c2, c3],
                    "pit_laps": [pit1, pit2],
                    "stint_lengths": [s1, s2, s3],
                    "stint_avgs": [
                        sum(t1) / len(t1),
                        sum(t2) / len(t2),
                        sum(t3) / len(t3),
                    ],
                    "stint_last": [t1[-1], t2[-1], t3[-1]],
                    "total_time": total_time,
                })

    results.sort(key=lambda x: x["total_time"])
    return results


def _simulate_live_strategy(circuit_name, current_lap, current_compound,
                            tyre_age, total_laps, pit_loss, track_temp,
                            driver, models):
    """Mid-race: project current stint, find pit window and best compound."""
    remaining = total_laps - current_lap

    # 1. Stay-out scenario (finish race on current tyres, no more pits)
    stay_out_times = _project_stint(
        circuit_name, current_compound, remaining,
        current_lap + 1, total_laps, track_temp, driver, models,
        tyre_life_start=tyre_age + 1,
    )
    stay_out_total = sum(stay_out_times)
    cliff_offset = _detect_cliff_lap(stay_out_times)

    # 2. Simulate pitting at each possible lap for each compound
    pit_options = []
    for new_compound in COMPOUNDS:
        for pit_offset in range(1, remaining - MIN_STINT_LAPS + 1,
                                PIT_SWEEP_STEP):
            pit_on_lap = current_lap + pit_offset
            laps_before = pit_offset
            laps_after = total_laps - pit_on_lap

            if laps_after < MIN_STINT_LAPS:
                continue

            # Current tyres until pit stop
            before_times = _project_stint(
                circuit_name, current_compound, laps_before,
                current_lap + 1, total_laps, track_temp, driver, models,
                tyre_life_start=tyre_age + 1,
            )
            # Fresh tyres after pit stop
            after_times = _project_stint(
                circuit_name, new_compound, laps_after,
                pit_on_lap + 1, total_laps, track_temp, driver, models,
                tyre_life_start=1,
            )

            total = sum(before_times) + pit_loss + sum(after_times)

            pit_options.append({
                "pit_lap": pit_on_lap,
                "new_compound": new_compound,
                "laps_before": laps_before,
                "laps_after": laps_after,
                "after_avg": sum(after_times) / len(after_times),
                "after_last": after_times[-1],
                "total_time": total,
                "delta_vs_stay": total - stay_out_total,
                "uses_different_compound": new_compound != current_compound,
            })

    pit_options.sort(key=lambda x: x["total_time"])
    return stay_out_times, stay_out_total, cliff_offset, pit_options


def _format_full_race(results, circuit_name, total_laps, pit_loss,
                      track_temp, driver, flag_label):
    """Format the pre-race simulation output."""
    if not results:
        return "No viable strategies found."

    best = results[0]
    best_time = best["total_time"]

    lines = [
        f"=== PIT WALL STRATEGY SIMULATION: {circuit_name} ({total_laps} laps) ===",
        f"  Driver: {driver} | Track Temp: {track_temp}C | "
        f"Pit Loss: {pit_loss:.1f}s ({flag_label})",
        "",
    ]

    # Optimal strategy detail
    compound_seq = " -> ".join(best["compounds"])
    pit_str = ", ".join([f"Lap {p}" for p in best["pit_laps"]])
    lines.append(
        f"OPTIMAL: {best['stops']}-Stop ({compound_seq})"
    )
    lines.append(
        f"  Pit on {pit_str} | "
        f"Total: {best_time:.1f}s ({best_time / 60:.1f} min)"
    )

    # Stint breakdown
    lap_cursor = 1
    for i, (comp, length, avg, last) in enumerate(zip(
        best["compounds"], best["stint_lengths"],
        best["stint_avgs"], best["stint_last"]
    )):
        end_lap = lap_cursor + length - 1
        lines.append(
            f"  Stint {i + 1}: {comp} (Laps {lap_cursor}-{end_lap}, "
            f"{length} laps) Avg: {avg:.2f}s, Last: {last:.2f}s"
        )
        lap_cursor = end_lap + 1

    # Alternative strategies (top 4 unique compound combos)
    lines.append("")
    lines.append("--- ALTERNATIVE STRATEGIES ---")
    seen = set()
    alt_count = 0
    for r in results[1:]:
        key = (r["stops"], tuple(r["compounds"]))
        if key in seen:
            continue
        seen.add(key)
        alt_count += 1
        if alt_count > 4:
            break

        seq = " -> ".join(r["compounds"])
        pits = ", ".join([f"L{p}" for p in r["pit_laps"]])
        delta = r["total_time"] - best_time
        rank = alt_count + 1
        suffix = "nd" if rank == 2 else "rd" if rank == 3 else "th"
        lines.append(
            f"  {rank}{suffix}: "
            f"{r['stops']}-Stop ({seq}) | Pits: {pits} | +{delta:.1f}s"
        )

    return "\n".join(lines)


def _format_live(stay_out_times, stay_out_total, cliff_offset,
                 pit_options, circuit_name, current_lap, current_compound,
                 tyre_age, total_laps, pit_loss, track_temp, driver,
                 flag_label):
    """Format the live mid-race strategy output."""
    remaining = total_laps - current_lap

    lines = [
        f"=== LIVE PIT STRATEGY: {circuit_name} — Lap {current_lap}/{total_laps} ===",
        f"  Current: {current_compound} tyres (Age: {tyre_age} laps) | "
        f"Track: {track_temp}C | Pit Loss: {pit_loss:.1f}s ({flag_label})",
        "",
    ]

    # Stay-out analysis
    lines.append(f"STAY OUT (No Pit): Total remaining: {stay_out_total:.1f}s")
    if cliff_offset is not None:
        cliff_race_lap = current_lap + cliff_offset
        lines.append(
            f"  WARNING: Tyre cliff detected at ~Lap {cliff_race_lap} "
            f"(degradation > {CLIFF_DEG_THRESHOLD}s/lap)"
        )
    lines.append("")

    # Pit recommendations
    if not pit_options:
        lines.append("No viable pit options (race nearly complete).")
        return "\n".join(lines)

    # Best option (prefer options that use a different compound)
    best_diff = [o for o in pit_options if o["uses_different_compound"]]
    best = best_diff[0] if best_diff else pit_options[0]

    saving = stay_out_total - best["total_time"]
    lines.append(
        f"RECOMMENDED: Box Lap {best['pit_lap']} -> {best['new_compound']}"
    )
    lines.append(
        f"  {best['new_compound']} stint ({best['laps_after']} laps): "
        f"Avg {best['after_avg']:.2f}s, Last: {best['after_last']:.2f}s"
    )
    lines.append(f"  Total remaining: {best['total_time']:.1f}s")
    if saving > 0:
        lines.append(f"  Saves {saving:.1f}s vs staying out")
    else:
        lines.append(
            f"  Costs {abs(saving):.1f}s vs staying out "
            f"(pit only if compound rule not yet met)"
        )

    # Pit window (laps within 2s of optimal pit timing for same compound)
    same_compound_opts = [
        o for o in pit_options
        if o["new_compound"] == best["new_compound"]
        and o["total_time"] <= best["total_time"] + 2.0
    ]
    if same_compound_opts:
        window_start = min(o["pit_lap"] for o in same_compound_opts)
        window_end = max(o["pit_lap"] for o in same_compound_opts)
        lines.append(f"  Pit Window: Lap {window_start}-{window_end}")

    # Alternatives (best of each other compound)
    lines.append("")
    lines.append("--- ALTERNATIVES ---")
    seen_compounds = {best["new_compound"]}
    for opt in pit_options:
        if opt["new_compound"] in seen_compounds:
            continue
        seen_compounds.add(opt["new_compound"])
        delta = opt["total_time"] - best["total_time"]
        cliff_warn = ""
        if opt["laps_after"] > 20 and opt["new_compound"] == "SOFT":
            cliff_warn = " (may cliff)"
        lines.append(
            f"  Box Lap {opt['pit_lap']} -> {opt['new_compound']}: "
            f"Total {opt['total_time']:.1f}s "
            f"(+{delta:.1f}s vs optimal){cliff_warn}"
        )

    return "\n".join(lines)


@tool
def calculate_strategy(
    circuit_name: str,
    current_lap: int = 1,
    total_laps: Optional[int] = None,
    flag_condition: str = "green",
    target_compound: str = "MEDIUM",
    base_lap_time: Optional[float] = None,
    driver: str = "VER",
    track_temp: Optional[float] = None,
    current_compound: Optional[str] = None,
    tyre_age: Optional[int] = None,
) -> str:
    """Calculate race strategy projections and optimal pit decisions.

    Operates in two modes:
    - PRE-RACE (current_lap <= 1): Simulates all 1-stop and 2-stop strategies,
      enforces FIA 2-compound rule, and ranks by total race time.
    - LIVE (current_lap > 1): Calculates optimal pit window, detects tyre cliff,
      and recommends the best compound to switch to.

    Args:
        circuit_name: Circuit/location name (e.g. 'Monza', 'Silverstone').
        current_lap: Current lap number (1 = pre-race simulation).
        total_laps: Total race laps (auto-fetched from DB if None).
        flag_condition: 'green', 'safety_car', or 'vsc'.
        target_compound: Fallback compound for basic math mode.
        base_lap_time: Optional override for base lap time (fallback only).
        driver: 3-letter driver code (e.g. 'VER', 'NOR').
        track_temp: Track surface temperature in Celsius.
        current_compound: Compound the driver is currently on (live mode).
        tyre_age: Laps completed on current tyre set (live mode).

    Returns:
        Formatted strategy analysis with optimal strategy, pit windows,
        and compound recommendations.
    """
    effective_track_temp = track_temp if track_temp is not None else 35.0
    logger.info(
        f"Strategy calculation: circuit={circuit_name}, "
        f"lap={current_lap}/{total_laps}, flag={flag_condition}, "
        f"track_temp={effective_track_temp}"
    )

    # ── Fetch circuit data from DB ──
    summaries = _query_db(
        "SELECT * FROM circuit_summaries WHERE circuit_name LIKE ?",
        (f"%{circuit_name}%",),
    )
    if not summaries:
        logger.warning(f"No telemetry data for circuit: {circuit_name}")
        return f"No telemetry data found for circuit matching '{circuit_name}'."

    summary = summaries[0]

    # Resolve total_laps
    if not total_laps:
        laps_row = _query_db(
            "SELECT total_laps FROM circuits WHERE circuit_name LIKE ? LIMIT 1",
            (f"%{circuit_name}%",),
        )
        if laps_row and laps_row[0]["total_laps"]:
            total_laps = int(laps_row[0]["total_laps"])
        else:
            total_laps = 55

    remaining_laps = total_laps - current_lap
    if remaining_laps <= 0:
        return "Race is already complete — no strategy calculation needed."

    # ── Pit loss calculation ──
    green_flag_loss = summary["avg_green_flag_pit_loss"] or 25.0
    flag_condition_lower = flag_condition.lower().replace(" ", "_")
    if flag_condition_lower == "safety_car":
        pit_loss = max(green_flag_loss - 12.0, 8.0)
        flag_label = "Safety Car"
    elif flag_condition_lower == "vsc":
        pit_loss = max(green_flag_loss - 7.0, 10.0)
        flag_label = "Virtual Safety Car"
    else:
        pit_loss = green_flag_loss
        flag_label = "Green Flag"

    # ── Load ML models ──
    models = _load_hybrid_models()

    if models is None:
        # Fallback to basic math if ML models are unavailable
        logger.info("ML models not available. Using basic math fallback.")
        compound_upper = (current_compound or target_compound or "MEDIUM").upper()
        deg_key = f"avg_{compound_upper.lower()}_deg_per_lap"
        deg_per_lap = summary.get(deg_key) or 0.05

        if base_lap_time is None:
            avg_rows = _query_db(
                "SELECT AVG(t.avg_lap_time_seconds) as avg_time "
                "FROM tyre_stints t JOIN circuits c ON t.circuit_id = c.id "
                "WHERE c.circuit_name LIKE ? AND t.compound = ?",
                (f"%{circuit_name}%", compound_upper),
            )
            if avg_rows and avg_rows[0]["avg_time"]:
                base_lap_time = avg_rows[0]["avg_time"]
            else:
                base_lap_time = 90.0

        lap_times = []
        total_stint_time = 0.0
        for lap_offset in range(remaining_laps):
            projected = base_lap_time + (deg_per_lap * lap_offset)
            total_stint_time += projected
            lap_times.append(projected)

        total_race_time = total_stint_time + pit_loss
        return (
            f"=== BASIC STRATEGY ESTIMATE: {summary['circuit_name']} ===\n"
            f"  (ML models not loaded — using database averages)\n\n"
            f"  Compound: {compound_upper} | Laps remaining: {remaining_laps}\n"
            f"  Base pace: {base_lap_time:.2f}s | Deg: {deg_per_lap:+.4f}s/lap\n"
            f"  Projected first: {lap_times[0]:.2f}s | Last: {lap_times[-1]:.2f}s\n"
            f"  Total stint: {total_stint_time:.1f}s | With pit: {total_race_time:.1f}s\n"
        )

    # ── Determine mode and run simulation ──
    circuit_db_name = summary["circuit_name"]  # Exact DB name for ML alignment

    if current_lap <= 1:
        # PRE-RACE: Full simulation of all strategies
        logger.info(f"Running PRE-RACE strategy simulation for {circuit_db_name}...")
        results = _simulate_full_race(
            circuit_db_name, total_laps, pit_loss,
            effective_track_temp, driver.upper(), models,
        )
        logger.info(f"Simulated {len(results)} strategy permutations.")
        return _format_full_race(
            results, circuit_db_name, total_laps, pit_loss,
            effective_track_temp, driver.upper(), flag_label,
        )
    else:
        # LIVE: Pit window + compound recommendation
        effective_compound = (current_compound or target_compound or "MEDIUM").upper()
        effective_tyre_age = tyre_age if tyre_age is not None else current_lap

        logger.info(
            f"Running LIVE strategy for {circuit_db_name}: "
            f"Lap {current_lap}, {effective_compound} (age {effective_tyre_age})"
        )
        stay_out_times, stay_out_total, cliff_offset, pit_options = (
            _simulate_live_strategy(
                circuit_db_name, current_lap, effective_compound,
                effective_tyre_age, total_laps, pit_loss,
                effective_track_temp, driver.upper(), models,
            )
        )
        logger.info(
            f"Live simulation: {len(pit_options)} pit options evaluated."
        )
        return _format_live(
            stay_out_times, stay_out_total, cliff_offset, pit_options,
            circuit_db_name, current_lap, effective_compound,
            effective_tyre_age, total_laps, pit_loss,
            effective_track_temp, driver.upper(), flag_label,
        )




