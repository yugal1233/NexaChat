"""Utility functions for NexaChat."""

import os
from datetime import datetime


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
