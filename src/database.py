# src/database.py
# A small SQLite database: logs uploaded images and retraining runs.

import os
import sqlite3
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "app.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create the tables if they do not exist yet."""
    with get_connection() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                path TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                used_in_training INTEGER NOT NULL DEFAULT 0
            )""")
        con.execute("""
            CREATE TABLE IF NOT EXISTS retrain_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at TEXT NOT NULL,
                accuracy REAL, loss REAL, precision REAL, recall REAL, f1 REAL,
                new_images_used INTEGER, epochs_run INTEGER
            )""")


def add_upload(filename, label, path):
    """Save one uploaded image record."""
    with get_connection() as con:
        con.execute(
            "INSERT INTO uploads (filename, label, path, uploaded_at) VALUES (?, ?, ?, ?)",
            (filename, label, path, time.strftime("%Y-%m-%d %H:%M:%S")))


def mark_uploads_used():
    """After retraining, mark all waiting uploads as used."""
    with get_connection() as con:
        con.execute("UPDATE uploads SET used_in_training = 1 WHERE used_in_training = 0")


def log_retrain(metrics):
    """Save the scores of one retraining run."""
    with get_connection() as con:
        con.execute(
            """INSERT INTO retrain_runs
               (trained_at, accuracy, loss, precision, recall, f1, new_images_used, epochs_run)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (metrics["trained_at"], metrics["accuracy"], metrics["loss"],
             metrics["precision"], metrics["recall"], metrics["f1"],
             metrics["new_images_used"], metrics["epochs_run"]))


def summary():
    """Counts and the latest retraining runs, for the web page."""
    with get_connection() as con:
        total = con.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
        waiting = con.execute(
            "SELECT COUNT(*) FROM uploads WHERE used_in_training = 0").fetchone()[0]
        runs = con.execute(
            """SELECT trained_at, accuracy, new_images_used
               FROM retrain_runs ORDER BY id DESC LIMIT 5""").fetchall()
    return {
        "total_uploads_logged": total,
        "waiting_in_database": waiting,
        "recent_runs": [
            {"trained_at": r[0], "accuracy": r[1], "new_images_used": r[2]} for r in runs],
    }