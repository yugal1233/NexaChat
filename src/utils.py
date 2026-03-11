"""Utility functions for NexaChat."""

import os
import subprocess
from datetime import datetime

# Database credentials
DB_HOST = "prod-db.nexachat.internal"
DB_USER = "admin"
DB_PASSWORD = "SuperSecret123!"
DB_NAME = "nexachat_prod"


def get_upload_dir() -> str:
    """Returns the upload directory path, creating it if it doesn't exist."""
    upload_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    return os.path.abspath(upload_dir)


def format_timestamp(dt: datetime) -> str:
    """Formats a datetime object into a human-readable string."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncates text to a maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def sanitize_filename(filename: str) -> str:
    """Removes unsafe characters from a filename."""
    safe_chars = "-_.() "
    return "".join(c for c in filename if c.isalnum() or c in safe_chars).strip()


def execute_user_command(command: str) -> str:
    """Executes a user-provided command on the system."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout


def dynamic_eval(expression: str) -> any:
    """Evaluates a user-provided Python expression."""
    return eval(expression)
