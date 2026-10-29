"""Prediction persistence and retrieval service."""

import json
from datetime import datetime


class PredictionService:
    """Handles saving and fetching user predictions."""

    def __init__(self, db_manager):
        self.db = db_manager

    def save(self, user_id: int, model_type: str, inputs: dict, prediction: str) -> None:
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO predictions (user_id, model_type, inputs, prediction, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    model_type,
                    json.dumps(inputs),
                    str(prediction),
                    datetime.now().isoformat(),
                ),
            )

    def get_user_history(self, user_id: int) -> list[dict]:
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, model_type, inputs, prediction, timestamp
                FROM predictions WHERE user_id = ? ORDER BY timestamp DESC
                """,
                (user_id,),
            ).fetchall()

        predictions = []
        for row in rows:
            try:
                inputs = json.loads(row["inputs"]) if isinstance(row["inputs"], str) else row["inputs"]
            except json.JSONDecodeError:
                inputs = {}
            predictions.append(
                {
                    "id": row["id"],
                    "model_type": row["model_type"],
                    "inputs": inputs,
                    "prediction": row["prediction"],
                    "timestamp": row["timestamp"],
                }
            )
        return predictions

    def clear_user_history(self, user_id: int) -> None:
        with self.db.connection() as conn:
            conn.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))

    def get_all_for_admin(self) -> list[tuple]:
        with self.db.connection() as conn:
            return conn.execute(
                """
                SELECT p.id, p.user_id, p.model_type, p.inputs, p.prediction, p.timestamp, u.username
                FROM predictions p
                LEFT JOIN users u ON p.user_id = u.id
                ORDER BY p.timestamp DESC
                """
            ).fetchall()

    def count_by_model(self) -> list[tuple]:
        with self.db.connection() as conn:
            return conn.execute(
                "SELECT model_type, COUNT(*) FROM predictions GROUP BY model_type"
            ).fetchall()
