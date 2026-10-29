"""Audit logging service."""

import json
from datetime import datetime

from flask import request


class AuditService:
    """Records user actions for security auditing."""

    def __init__(self, db_manager):
        self.db = db_manager

    def log(self, user_id: int | None, action: str, details: dict | None = None) -> None:
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_log (user_id, action, details, ip_address, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    action,
                    json.dumps(details) if details else None,
                    request.remote_addr if request else None,
                    datetime.now().isoformat(),
                ),
            )
