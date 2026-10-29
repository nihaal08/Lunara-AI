"""SQLite database manager and schema initialization."""

import sqlite3
from contextlib import contextmanager
from typing import Generator


class DatabaseManager:
    """Manages SQLite connections and schema setup."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._shared_conn = None
        self._mem_uri = None
        if db_path == ":memory:":
            self._mem_uri = f"file:lunara_mem_{id(self)}?mode=memory&cache=shared"
            self._shared_conn = sqlite3.connect(self._mem_uri, uri=True, check_same_thread=False)
            self._shared_conn.execute("PRAGMA foreign_keys = ON")
        else:
            from pathlib import Path
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        if self._mem_uri:
            conn = sqlite3.connect(self._mem_uri, uri=True, check_same_thread=False)
        else:
            conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for safe database access."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create application tables if they do not exist."""
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    is_admin INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_login TEXT,
                    is_active INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    model_type TEXT NOT NULL,
                    inputs TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    details TEXT,
                    ip_address TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                """
            )
