"""Logging configuration for LUNARA-AI."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(app) -> None:
    """Configure application-wide logging."""
    log_level = getattr(logging, app.config.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    app.logger.setLevel(log_level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    app.logger.addHandler(console_handler)

    # File handler (skip for in-memory testing)
    if app.config["DATABASE_PATH"] != ":memory:":
        log_file = Path(app.config["LOG_FILE"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
