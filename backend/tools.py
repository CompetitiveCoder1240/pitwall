"""
PitWall — LangChain Tool Definitions
=====================================
Defines @tool functions that the LangGraph Agent can call:

1. query_telemetry  — Fast SQLite lookup for pit stop & tyre degradation data
2. calculate_strategy — Deterministic Python calculator for race stint projections
"""

import sqlite3
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Database path — resolved relative to project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "pitwall_telemetry.db"


def _query_db(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a read-only SQL query and return results as list of dicts."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(sql, params)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


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
    if query_type == "all_circuits":
        rows = _query_db(
            "SELECT circuit_name, total_races_sampled, avg_pit_duration, "
            "avg_soft_deg_per_lap, avg_medium_deg_per_lap, avg_hard_deg_per_lap "
            "FROM circuit_summaries ORDER BY circuit_name"
        )
        if not rows:
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

    return f"Unknown query_type '{query_type}'. Use: summary, pit_stops, tyre_stints, or all_circuits."


# ---------------------------------------------------------------------------
# Tool 2: Race Strategy Calculator
# ---------------------------------------------------------------------------
@tool
def calculate_strategy(
    circuit_name: str,
    current_lap: int,
    total_laps: int,
    flag_condition: str = "green",
    target_compound: str = "MEDIUM",
    base_lap_time: Optional[float] = None,
) -> str:
    """Calculate race strategy projections for a pit stop decision.

    Uses telemetry data from the PitWall database to compute:
    - Pit stop time loss under current flag conditions
    - Projected stint degradation for the target tyre compound
    - Total remaining race time estimate

    Args:
        circuit_name: The circuit/location name (e.g. 'Monza', 'Silverstone').
        current_lap: The current lap number.
        total_laps: The total number of laps in the race.
        flag_condition: Current flag state - 'green', 'safety_car', or 'vsc'.
        target_compound: Tyre compound to switch to - 'SOFT', 'MEDIUM', or 'HARD'.
        base_lap_time: Optional override for base lap time in seconds.
                       If not provided, uses the circuit average from the database.

    Returns:
        Formatted strategy analysis with pit loss, stint projection, and recommendation.
    """
    remaining_laps = total_laps - current_lap
    if remaining_laps <= 0:
        return "Race is already complete — no strategy calculation needed."

    # Fetch circuit summary from DB
    summaries = _query_db(
        "SELECT * FROM circuit_summaries WHERE circuit_name LIKE ?",
        (f"%{circuit_name}%",),
    )

    if not summaries:
        return f"No telemetry data found for circuit matching '{circuit_name}'."

    summary = summaries[0]

    # --- Pit loss calculation ---
    green_flag_loss = summary["avg_green_flag_pit_loss"] or 25.0

    flag_condition_lower = flag_condition.lower().replace(" ", "_")
    if flag_condition_lower == "safety_car":
        pit_loss = max(green_flag_loss - 12.0, 8.0)
        flag_label = "Safety Car"
        time_saved_vs_green = green_flag_loss - pit_loss
    elif flag_condition_lower == "vsc":
        pit_loss = max(green_flag_loss - 7.0, 10.0)
        flag_label = "Virtual Safety Car"
        time_saved_vs_green = green_flag_loss - pit_loss
    else:
        pit_loss = green_flag_loss
        flag_label = "Green Flag"
        time_saved_vs_green = 0.0

    # --- Tyre degradation lookup ---
    compound_upper = target_compound.upper()
    deg_key = f"avg_{compound_upper.lower()}_deg_per_lap"
    deg_per_lap = summary.get(deg_key)
    if deg_per_lap is None:
        deg_per_lap = 0.05  # Default fallback

    # --- Base lap time ---
    if base_lap_time is None:
        # Fetch average lap time from tyre stints for this circuit
        avg_rows = _query_db(
            "SELECT AVG(t.avg_lap_time_seconds) as avg_time "
            "FROM tyre_stints t JOIN circuits c ON t.circuit_id = c.id "
            "WHERE c.circuit_name LIKE ? AND t.compound = ?",
            (f"%{circuit_name}%", compound_upper),
        )
        if avg_rows and avg_rows[0]["avg_time"]:
            base_lap_time = avg_rows[0]["avg_time"]
        else:
            base_lap_time = 90.0  # Default fallback

    # --- Stint projection ---
    total_stint_time = 0.0
    lap_times = []
    for lap_offset in range(remaining_laps):
        projected_time = base_lap_time + (deg_per_lap * lap_offset)
        total_stint_time += projected_time
        lap_times.append(projected_time)

    total_race_time = total_stint_time + pit_loss

    # --- Tyre cliff warning ---
    cliff_warning = ""
    if compound_upper == "SOFT" and remaining_laps > 20:
        cliff_warning = "\n  ⚠️ WARNING: Soft tyres may hit performance cliff before lap end (typical limit ~18-22 laps)."
    elif compound_upper == "MEDIUM" and remaining_laps > 35:
        cliff_warning = "\n  ⚠️ WARNING: Medium tyres may degrade significantly over 35+ laps."

    # --- Format output ---
    result = (
        f"═══ PIT STRATEGY ANALYSIS: {summary['circuit_name']} ═══\n\n"
        f"  Current Lap: {current_lap} / {total_laps} ({remaining_laps} laps remaining)\n"
        f"  Flag Condition: {flag_label}\n"
        f"  Target Compound: {compound_upper}\n\n"
        f"─── PIT STOP COST ───\n"
        f"  Green-flag pit loss: {green_flag_loss:.1f}s\n"
        f"  {flag_label} pit loss: {pit_loss:.1f}s\n"
    )
    if time_saved_vs_green > 0:
        result += f"  Time saved vs green-flag pit: {time_saved_vs_green:.1f}s\n"

    result += (
        f"\n─── STINT PROJECTION ({compound_upper}, {remaining_laps} laps) ───\n"
        f"  Base lap time: {base_lap_time:.2f}s\n"
        f"  Degradation rate: {deg_per_lap:+.4f}s per lap\n"
        f"  Projected first lap: {lap_times[0]:.2f}s\n"
        f"  Projected last lap: {lap_times[-1]:.2f}s\n"
        f"  Total stint time: {total_stint_time:.1f}s ({total_stint_time / 60:.1f} min)\n"
        f"  Total time (stint + pit): {total_race_time:.1f}s ({total_race_time / 60:.1f} min)\n"
    )

    if cliff_warning:
        result += cliff_warning

    return result
