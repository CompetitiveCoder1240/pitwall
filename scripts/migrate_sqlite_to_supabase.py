import sqlite3
import os
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USERS_DB = PROJECT_ROOT / "data" / "users.db"
TELEMETRY_DB = PROJECT_ROOT / "data" / "pitwall_telemetry.db"

load_dotenv(PROJECT_ROOT / ".env")
SUPABASE_URL = os.getenv("SUPABASE_URL")

def create_postgres_tables(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS circuits (
            id SERIAL PRIMARY KEY,
            circuit_name VARCHAR(255) NOT NULL,
            season INTEGER NOT NULL,
            event_name VARCHAR(255)
        );
        
        CREATE TABLE IF NOT EXISTS pit_stop_deltas (
            id SERIAL PRIMARY KEY,
            circuit_id INTEGER REFERENCES circuits(id),
            driver VARCHAR(100),
            lap_number INTEGER,
            pit_duration_seconds DOUBLE PRECISION,
            compound_before VARCHAR(50),
            compound_after VARCHAR(50)
        );
        
        CREATE TABLE IF NOT EXISTS tyre_stints (
            id SERIAL PRIMARY KEY,
            circuit_id INTEGER REFERENCES circuits(id),
            driver VARCHAR(100),
            stint_number INTEGER,
            compound VARCHAR(50),
            start_lap INTEGER,
            end_lap INTEGER,
            stint_length INTEGER,
            avg_lap_time_seconds DOUBLE PRECISION,
            degradation_per_lap DOUBLE PRECISION
        );
        
        CREATE TABLE IF NOT EXISTS circuit_summaries (
            id SERIAL PRIMARY KEY,
            circuit_name VARCHAR(255) UNIQUE NOT NULL,
            total_races_sampled INTEGER,
            avg_pit_duration DOUBLE PRECISION,
            avg_green_flag_pit_loss DOUBLE PRECISION,
            avg_soft_deg_per_lap DOUBLE PRECISION,
            avg_medium_deg_per_lap DOUBLE PRECISION,
            avg_hard_deg_per_lap DOUBLE PRECISION
        );
    """)
    conn.commit()

def migrate_table(sqlite_db, table_name, pg_conn, columns):
    print(f"Migrating {table_name}...")
    sqlite_conn = sqlite3.connect(sqlite_db)
    sqlite_cursor = sqlite_conn.cursor()
    sqlite_cursor.execute(f"SELECT {', '.join(columns)} FROM {table_name}")
    rows = sqlite_cursor.fetchall()
    sqlite_conn.close()

    if not rows:
        print(f"No data in {table_name}.")
        return

    pg_cursor = pg_conn.cursor()
    pg_cursor.execute(f"TRUNCATE TABLE {table_name} CASCADE")
    
    insert_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES %s"
    execute_values(pg_cursor, insert_query, rows)
    pg_conn.commit()
    print(f"Migrated {len(rows)} rows to {table_name}.")

if __name__ == "__main__":
    if not SUPABASE_URL:
        print("ERROR: SUPABASE_URL not found in environment.")
        exit(1)

    print("Connecting to Supabase...")
    pg_conn = psycopg2.connect(SUPABASE_URL)

    print("Creating tables if they don't exist...")
    create_postgres_tables(pg_conn)

    if USERS_DB.exists():
        migrate_table(USERS_DB, "users", pg_conn, ["username", "password_hash"])
    
    if TELEMETRY_DB.exists():
        migrate_table(TELEMETRY_DB, "circuits", pg_conn, ["id", "circuit_name", "season", "event_name"])
        migrate_table(TELEMETRY_DB, "pit_stop_deltas", pg_conn, ["circuit_id", "driver", "lap_number", "pit_duration_seconds", "compound_before", "compound_after"])
        migrate_table(TELEMETRY_DB, "tyre_stints", pg_conn, ["circuit_id", "driver", "stint_number", "compound", "start_lap", "end_lap", "stint_length", "avg_lap_time_seconds", "degradation_per_lap"])
        migrate_table(TELEMETRY_DB, "circuit_summaries", pg_conn, ["circuit_name", "total_races_sampled", "avg_pit_duration", "avg_green_flag_pit_loss", "avg_soft_deg_per_lap", "avg_medium_deg_per_lap", "avg_hard_deg_per_lap"])

    pg_conn.close()
    print("Migration complete!")
