"""User authentication service."""

from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash


class AuthService:
    """Handles user registration, login, and admin user management."""

    def __init__(self, db_manager):
        self.db = db_manager

    def create_user(self, username: str, full_name: str, email: str, password: str) -> None:
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO users (username, full_name, email, password, is_admin, created_at)
                VALUES (?, ?, ?, ?, 0, ?)
                """,
                (
                    username,
                    full_name,
                    email,
                    generate_password_hash(password),
                    datetime.now().isoformat(),
                ),
            )

    def authenticate(self, identifier: str, password: str) -> dict | None:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT id, username, email, password, is_admin
                FROM users
                WHERE (username = ? OR email = ?) AND is_active = 1
                """,
                (identifier, identifier),
            ).fetchone()

        if row and check_password_hash(row["password"], password):
            with self.db.connection() as conn:
                conn.execute(
                    "UPDATE users SET last_login = ? WHERE id = ?",
                    (datetime.now().isoformat(), row["id"]),
                )
            return {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "is_admin": bool(row["is_admin"]),
            }
        return None

    def get_all_users(self) -> list[tuple]:
        with self.db.connection() as conn:
            return conn.execute(
                """
                SELECT id, username, full_name, email, password, is_admin, created_at
                FROM users ORDER BY id
                """
            ).fetchall()

    def delete_user(self, user_id: int) -> None:
        with self.db.connection() as conn:
            conn.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM audit_log WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def count_users(self) -> int:
        with self.db.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def count_predictions(self) -> int:
        with self.db.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def count_active_users(self) -> int:
        with self.db.connection() as conn:
            return conn.execute(
                "SELECT COUNT(DISTINCT user_id) FROM predictions"
            ).fetchone()[0]
