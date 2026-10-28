"""
ClashVision Storage Subsystem — SQLite WAL Connection Manager.
Provides thread-safe connection pooling, automatic schema initialization,
WAL (Write-Ahead Logging) optimization, and transaction safety.
Derived from Requirement R5 and Feature 38.
"""

from __future__ import annotations
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple


SCHEMA_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")

EXPECTED_TABLES = [
    "sessions",
    "base_scans",
    "decisions",
    "evaluations",
    "settings",
    "logs",
    "dataset_annotations",
]


class DatabaseManager:
    """
    SQLite database connection manager configured for WAL mode and high-concurrency read/write.
    Guarantees:
    - PRAGMA journal_mode=WAL; (for file-based databases)
    - PRAGMA synchronous=NORMAL;
    - PRAGMA foreign_keys=ON;
    - PRAGMA busy_timeout=5000;
    - Thread-safe connection handling using thread-local storage.
    - Automatic schema bootstrapping from schema.sql.
    """

    def __init__(self, db_path: str = "clashvision.db", auto_init: bool = True) -> None:
        self.db_path = db_path
        self._is_memory = (db_path == ":memory:" or (db_path.startswith("file:") and "mode=memory" in db_path))
        self._memory_lock = threading.RLock()
        self._local = threading.local()
        self._connections_lock = threading.Lock()
        self._all_connections: List[sqlite3.Connection] = []
        self._shared_memory_conn: Optional[sqlite3.Connection] = None
        self._is_closed = False

        if self._is_memory:
            # For in-memory databases, maintain a primary anchor connection so tables persist
            self._shared_memory_conn = self._create_raw_connection()

        if auto_init:
            self.init_schema()

    def _create_raw_connection(self) -> sqlite3.Connection:
        """Create and configure a new sqlite3 connection with WAL pragmas."""
        if not self._is_memory and self.db_path != ":memory:":
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(
            self.db_path,
            timeout=10.0,
            check_same_thread=False,
            isolation_level=None,  # autocommit mode; transactions managed via BEGIN/COMMIT
        )
        conn.row_factory = sqlite3.Row

        # Apply database pragmas
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("PRAGMA busy_timeout = 5000;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
        if not self._is_memory:
            cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.close()

        with self._connections_lock:
            self._all_connections.append(conn)

        return conn

    def get_connection(self) -> sqlite3.Connection:
        """Retrieve thread-local connection, or shared connection for memory mode."""
        if self._is_closed:
            raise sqlite3.ProgrammingError("Cannot operate on a closed database.")

        if self._is_memory and self._shared_memory_conn is not None:
            return self._shared_memory_conn

        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._create_raw_connection()
            self._local.conn = conn
        return conn

    def init_schema(self, schema_file: Optional[str] = None) -> None:
        """Execute schema.sql to ensure all 7 tables and indexes exist."""
        path = schema_file or SCHEMA_FILE_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Schema definition file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        if self._is_memory:
            with self._memory_lock:
                conn = self.get_connection()
                conn.executescript(schema_sql)
        else:
            conn = self.get_connection()
            conn.executescript(schema_sql)

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing explicit atomic transaction boundary."""
        if self._is_memory:
            with self._memory_lock:
                conn = self.get_connection()
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE;")
                try:
                    yield conn
                    if conn.in_transaction:
                        cursor.execute("COMMIT;")
                except Exception:
                    if conn.in_transaction:
                        cursor.execute("ROLLBACK;")
                    raise
                finally:
                    cursor.close()
        else:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE;")
            try:
                yield conn
                if conn.in_transaction:
                    cursor.execute("COMMIT;")
            except Exception:
                if conn.in_transaction:
                    cursor.execute("ROLLBACK;")
                raise
            finally:
                cursor.close()

    def execute(self, sql: str, params: Tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """Execute a single SQL statement."""
        if self._is_memory:
            with self._memory_lock:
                conn = self.get_connection()
                return conn.execute(sql, params)
        conn = self.get_connection()
        return conn.execute(sql, params)

    def fetchone(self, sql: str, params: Tuple[Any, ...] = ()) -> Optional[sqlite3.Row]:
        """Execute a query and fetch a single row."""
        cursor = self.execute(sql, params)
        return cursor.fetchone()

    def fetchall(self, sql: str, params: Tuple[Any, ...] = ()) -> List[sqlite3.Row]:
        """Execute a query and fetch all rows."""
        cursor = self.execute(sql, params)
        return cursor.fetchall()

    def get_table_names(self) -> List[str]:
        """List all user tables in the database."""
        rows = self.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        )
        return [row["name"] for row in rows]

    def table_exists(self, table_name: str) -> bool:
        """Check whether a specific table exists."""
        row = self.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?;",
            (table_name,),
        )
        return row is not None

    def check_wal_mode(self) -> str:
        """Query and return the current SQLite journal mode."""
        row = self.fetchone("PRAGMA journal_mode;")
        if row:
            return str(row[0]).lower()
        return "unknown"

    def verify_schema_integrity(self) -> bool:
        """Verify that all 7 required tables are present."""
        existing_tables = set(self.get_table_names())
        return all(tbl in existing_tables for tbl in EXPECTED_TABLES)

    def close(self) -> None:
        """Close all active connections."""
        self._is_closed = True
        with self._connections_lock:
            for conn in self._all_connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_connections.clear()
            self._shared_memory_conn = None
            if hasattr(self._local, "conn"):
                del self._local.conn
