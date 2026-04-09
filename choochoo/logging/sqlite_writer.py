"""Async SQLite metrics writer for training runs.

Writes are non-blocking: log() enqueues a row and returns immediately.
A background daemon thread drains the queue and does the actual INSERT.

DB filename convention: <model>-<name>.db  (mirrors safetensors naming)
"""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TrainingPhase(str, Enum):
    WARMUP      = "warmup"
    BURNIN      = "burnin"
    TRAINING    = "training"
    CONVERGENCE = "convergence"


_STOP = object()  # sentinel to shut down the worker thread


class SQLiteMetricsWriter:
    """Non-blocking SQLite metrics sink.

    Usage::

        writer = SQLiteMetricsWriter("/output/qwen-myrun.db")
        writer.log(step=1, loss=0.42, lr=1e-4, phase=TrainingPhase.WARMUP,
                   samples_per_sec=2.1, gpu_util=0.93)
        writer.close()
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="sqlite-metrics")
        self._thread.start()
        logger.info("SQLite metrics DB: %s", db_path)

    def log(
        self,
        step: int,
        loss: float,
        lr: float,
        phase: TrainingPhase,
        **extras: Any,
    ) -> None:
        """Enqueue a metrics row. Returns immediately (non-blocking)."""
        self._queue.put((step, float(loss), float(lr), phase.value, extras))

    def close(self) -> None:
        """Flush remaining writes and shut down the background thread."""
        self._queue.put(_STOP)
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            logger.warning("SQLiteMetricsWriter: worker thread did not exit cleanly")

    def _worker(self) -> None:
        con = sqlite3.connect(self._db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                step     INTEGER PRIMARY KEY,
                loss     REAL NOT NULL,
                lr       REAL NOT NULL,
                phase    TEXT NOT NULL,
                extras   TEXT          -- JSON: samples_per_sec, gpu_util, etc.
            )
        """)
        con.commit()

        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            step, loss, lr, phase, extras = item
            try:
                con.execute(
                    "INSERT OR REPLACE INTO metrics VALUES (?,?,?,?,?)",
                    (step, loss, lr, phase, json.dumps(extras)),
                )
                con.commit()
            except Exception:
                logger.exception("SQLiteMetricsWriter: failed to write step %d", step)

        con.close()
