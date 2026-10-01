import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager

from backend.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPABASE_URL = os.getenv("SUPABASE_URL")
LOCAL_USERS_DB_PATH = PROJECT_ROOT / "data" / "users.db"
LOCAL_TELEMETRY_DB_PATH = PROJECT_ROOT / "data" / "pitwall_telemetry.db"

_pg_pool = None

def get_pg_pool():
    global _pg_pool
    if _pg_pool is None and SUPABASE_URL:
        import psycopg2
        from psycopg2.pool import ThreadedConnectionPool
        try:
            _pg_pool = ThreadedConnectionPool(1, 20, dsn=SUPABASE_URL)
            logger.info("Initialized PostgreSQL connection pool.")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise e
    return _pg_pool

@contextmanager
def get_db_connection(db_type="users"):
    """
    Context manager for database connections.
    If SUPABASE_URL is configured, fetches a connection from the ThreadedConnectionPool.
    Otherwise, falls back to a local SQLite database (routing based on db_type).
    """
    if SUPABASE_URL:
        pool = get_pg_pool()
        conn = pool.getconn()
        try:
            yield conn
        finally:
            # Important: always return the connection to the pool
            pool.putconn(conn)
    else:
        db_path = LOCAL_USERS_DB_PATH if db_type == "users" else LOCAL_TELEMETRY_DB_PATH
        conn = sqlite3.connect(str(db_path))
        try:
            yield conn
        finally:
            conn.close()

def execute_query(sql: str, params: tuple = (), db_type="telemetry", fetch_one=False, commit=False):
    is_write = commit or sql.strip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"))
    
    if SUPABASE_URL:
        from psycopg2.extras import RealDictCursor
        pg_sql = sql.replace("?", "%s")
        with get_db_connection(db_type) as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(pg_sql, params)
            if is_write:
                conn.commit()
                return []
            if fetch_one:
                result = cursor.fetchone()
                if result:
                    return dict(result)
                return None
            return [dict(row) for row in cursor.fetchall()]
    else:
        with get_db_connection(db_type) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql, params)
            if is_write:
                conn.commit()
                return []
            if fetch_one:
                row = cursor.fetchone()
                return dict(row) if row else None
            return [dict(row) for row in cursor.fetchall()]
