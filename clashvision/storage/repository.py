"""
ClashVision Storage Subsystem — Data Repository Layer.
Provides high-level typed CRUD operations for Sessions, Base Scans,
Decisions, Evaluations, Settings, Logs, and Dataset Annotations.
Derived from Requirement R5 and Feature 38.
"""

from __future__ import annotations
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from clashvision.core.constants import GameState, StrategyType, RiskLevel
from clashvision.core.models import (
    SessionRecord,
    DatasetAnnotation,
    FarmingRecommendation,
    StrategyEvaluation,
)
from clashvision.storage.db import DatabaseManager


def _sanitize_str(val: Any) -> str:
    """Sanitize surrogate characters and normalize non-string types safely to UTF-8."""
    if val is None:
        return ""
    if not isinstance(val, str):
        val = str(val)
    return val.encode("utf-8", errors="replace").decode("utf-8")


class Repository:
    """
    Unified Data Access Layer for ClashVision persistence.
    Wraps DatabaseManager and provides typed domain CRUD methods.
    """

    def __init__(self, db: Optional[DatabaseManager] = None, db_path: str = "clashvision.db") -> None:
        self.db = db if db is not None else DatabaseManager(db_path=db_path)

    # =========================================================================
    # 1. Sessions CRUD
    # =========================================================================

    def create_session(
        self,
        session_id: Optional[str] = None,
        start_time: Optional[float] = None,
        notes: str = "",
    ) -> SessionRecord:
        """Create and persist a new application tracking session."""
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        st = start_time if start_time is not None else time.time()

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, start_time, end_time, total_scans, matched_targets, notes)
                VALUES (?, ?, NULL, 0, 0, ?);
                """,
                (sid, st, notes),
            )

        return SessionRecord(
            session_id=sid,
            start_time=st,
            end_time=None,
            total_scans=0,
            matched_targets=0,
            notes=notes,
        )

    def end_session(
        self,
        session_id: str,
        end_time: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> Optional[SessionRecord]:
        """Mark a session as terminated and record its finish timestamp."""
        et = end_time if end_time is not None else time.time()

        with self.db.transaction() as conn:
            if notes is not None:
                conn.execute(
                    "UPDATE sessions SET end_time = ?, notes = ? WHERE session_id = ?;",
                    (et, notes, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET end_time = ? WHERE session_id = ?;",
                    (et, session_id),
                )

        return self.get_session(session_id)

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Retrieve a session by unique identifier."""
        row = self.db.fetchone(
            "SELECT session_id, start_time, end_time, total_scans, matched_targets, notes FROM sessions WHERE session_id = ?;",
            (session_id,),
        )
        if not row:
            return None

        return SessionRecord(
            session_id=row["session_id"],
            start_time=float(row["start_time"]),
            end_time=float(row["end_time"]) if row["end_time"] is not None else None,
            total_scans=int(row["total_scans"] or 0),
            matched_targets=int(row["matched_targets"] or 0),
            notes=row["notes"] or "",
        )

    def list_sessions(self, limit: int = 100, offset: int = 0) -> List[SessionRecord]:
        """List historical sessions ordered newest first."""
        rows = self.db.fetchall(
            """
            SELECT session_id, start_time, end_time, total_scans, matched_targets, notes
            FROM sessions
            ORDER BY start_time DESC
            LIMIT ? OFFSET ?;
            """,
            (limit, offset),
        )
        return [
            SessionRecord(
                session_id=r["session_id"],
                start_time=float(r["start_time"]),
                end_time=float(r["end_time"]) if r["end_time"] is not None else None,
                total_scans=int(r["total_scans"] or 0),
                matched_targets=int(r["matched_targets"] or 0),
                notes=r["notes"] or "",
            )
            for r in rows
        ]

    def increment_session_scans(self, session_id: str, matched: bool = False) -> None:
        """Atomically increment scan counts for an active session."""
        with self.db.transaction() as conn:
            if matched:
                conn.execute(
                    """
                    UPDATE sessions
                    SET total_scans = total_scans + 1,
                        matched_targets = matched_targets + 1
                    WHERE session_id = ?;
                    """,
                    (session_id,),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET total_scans = total_scans + 1 WHERE session_id = ?;",
                    (session_id,),
                )

    # =========================================================================
    # 2. Base Scans CRUD
    # =========================================================================

    def save_base_scan(
        self,
        session_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        game_state: Union[str, GameState] = GameState.UNKNOWN,
        confidence: float = 1.0,
        gold: Optional[int] = None,
        elixir: Optional[int] = None,
        dark_elixir: Optional[int] = None,
        th_accessible: bool = False,
        raw_ocr_json: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> int:
        """Persist a captured base scan and update session scan counter."""
        ts = timestamp if timestamp is not None else time.time()
        st_val = _sanitize_str(game_state.value if isinstance(game_state, GameState) else str(game_state))
        if isinstance(raw_ocr_json, (dict, list)):
            ocr_str = json.dumps(raw_ocr_json)
        elif raw_ocr_json is not None:
            ocr_str = str(raw_ocr_json)
        else:
            ocr_str = ""
        ocr_str = _sanitize_str(ocr_str)
        th_int = 1 if th_accessible else 0
        sid = session_id.strip() if (isinstance(session_id, str) and session_id.strip()) else None

        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO base_scans (
                    session_id, timestamp, game_state, confidence,
                    gold, elixir, dark_elixir, th_accessible, raw_ocr_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (sid, ts, st_val, confidence, gold, elixir, dark_elixir, th_int, ocr_str),
            )
            scan_id = int(cursor.lastrowid)

            if sid:
                conn.execute(
                    "UPDATE sessions SET total_scans = total_scans + 1 WHERE session_id = ?;",
                    (sid,),
                )

        return scan_id

    def get_base_scans(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent base scans optionally filtered by session."""
        if session_id:
            rows = self.db.fetchall(
                """
                SELECT scan_id, session_id, timestamp, game_state, confidence,
                       gold, elixir, dark_elixir, th_accessible, raw_ocr_json
                FROM base_scans
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?;
                """,
                (session_id, limit, offset),
            )
        else:
            rows = self.db.fetchall(
                """
                SELECT scan_id, session_id, timestamp, game_state, confidence,
                       gold, elixir, dark_elixir, th_accessible, raw_ocr_json
                FROM base_scans
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?;
                """,
                (limit, offset),
            )

        results = []
        for r in rows:
            results.append({
                "scan_id": r["scan_id"],
                "session_id": r["session_id"],
                "timestamp": float(r["timestamp"]),
                "game_state": r["game_state"],
                "confidence": float(r["confidence"] or 0.0),
                "gold": r["gold"],
                "elixir": r["elixir"],
                "dark_elixir": r["dark_elixir"],
                "th_accessible": bool(r["th_accessible"]),
                "raw_ocr_json": r["raw_ocr_json"],
            })
        return results

    def get_scan_count(self, session_id: Optional[str] = None) -> int:
        """Count total recorded scans."""
        if session_id:
            row = self.db.fetchone("SELECT COUNT(*) FROM base_scans WHERE session_id = ?;", (session_id,))
        else:
            row = self.db.fetchone("SELECT COUNT(*) FROM base_scans;")
        return int(row[0]) if row else 0

    # =========================================================================
    # 3. Decisions & Strategy Evaluations CRUD
    # =========================================================================

    def save_recommendation(
        self,
        recommendation: FarmingRecommendation,
        session_id: Optional[str] = None,
        scan_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """
        Persist a complete FarmingRecommendation and all child StrategyEvaluations.
        Updates session matched_targets counter if recommendation is viable.
        """
        ts = timestamp if timestamp is not None else (recommendation.timestamp or time.time())
        strat_val = _sanitize_str(
            recommendation.primary_strategy.value
            if isinstance(recommendation.primary_strategy, StrategyType)
            else str(recommendation.primary_strategy)
        )
        explanation_str = _sanitize_str(recommendation.explanation or "")
        breakdown_str = _sanitize_str(json.dumps(recommendation.score_breakdown or {}))
        sid = session_id.strip() if (isinstance(session_id, str) and session_id.strip()) else None

        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO decisions (
                    scan_id, session_id, timestamp, primary_strategy,
                    suitability_score, explanation, breakdown_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    scan_id,
                    sid,
                    ts,
                    strat_val,
                    recommendation.suitability_score,
                    explanation_str,
                    breakdown_str,
                ),
            )
            decision_id = int(cursor.lastrowid)

            # Persist each candidate strategy evaluation
            for eval_item in recommendation.rankings:
                eval_strat = _sanitize_str(
                    eval_item.strategy_type.value
                    if isinstance(eval_item.strategy_type, StrategyType)
                    else str(eval_item.strategy_type)
                )
                eval_risk = _sanitize_str(
                    eval_item.risk_level.value
                    if isinstance(eval_item.risk_level, RiskLevel)
                    else str(eval_item.risk_level)
                )
                reasons_str = _sanitize_str(json.dumps(eval_item.reasons or []))

                cursor.execute(
                    """
                    INSERT INTO evaluations (
                        decision_id, strategy_type, score, viable, risk_level, reasons_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        decision_id,
                        eval_strat,
                        eval_item.score,
                        1 if eval_item.viable else 0,
                        eval_risk,
                        reasons_str,
                    ),
                )

            # If session is provided and base is suitable, increment matched targets
            if sid and recommendation.suitability_score >= 50.0:
                conn.execute(
                    "UPDATE sessions SET matched_targets = matched_targets + 1 WHERE session_id = ?;",
                    (sid,),
                )

        return decision_id

    def get_recent_decisions(
        self,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent tactical decisions with nested evaluations."""
        if session_id:
            rows = self.db.fetchall(
                """
                SELECT decision_id, scan_id, session_id, timestamp,
                       primary_strategy, suitability_score, explanation, breakdown_json
                FROM decisions
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?;
                """,
                (session_id, limit),
            )
        else:
            rows = self.db.fetchall(
                """
                SELECT decision_id, scan_id, session_id, timestamp,
                       primary_strategy, suitability_score, explanation, breakdown_json
                FROM decisions
                ORDER BY timestamp DESC
                LIMIT ?;
                """,
                (limit,),
            )

        decisions = []
        for r in rows:
            dec_id = r["decision_id"]
            eval_rows = self.db.fetchall(
                """
                SELECT eval_id, strategy_type, score, viable, risk_level, reasons_json
                FROM evaluations
                WHERE decision_id = ?
                ORDER BY score DESC;
                """,
                (dec_id,),
            )

            evals = []
            for er in eval_rows:
                reasons = []
                if er["reasons_json"]:
                    try:
                        reasons = json.loads(er["reasons_json"])
                    except Exception:
                        reasons = []
                evals.append({
                    "eval_id": er["eval_id"],
                    "strategy_type": er["strategy_type"],
                    "score": float(er["score"]),
                    "viable": bool(er["viable"]),
                    "risk_level": er["risk_level"],
                    "reasons": reasons,
                })

            breakdown = {}
            if r["breakdown_json"]:
                try:
                    breakdown = json.loads(r["breakdown_json"])
                except Exception:
                    breakdown = {}

            decisions.append({
                "decision_id": dec_id,
                "scan_id": r["scan_id"],
                "session_id": r["session_id"],
                "timestamp": float(r["timestamp"]),
                "primary_strategy": r["primary_strategy"],
                "suitability_score": float(r["suitability_score"] or 0.0),
                "explanation": r["explanation"] or "",
                "score_breakdown": breakdown,
                "evaluations": evals,
            })

        return decisions

    # =========================================================================
    # 4. Settings CRUD
    # =========================================================================

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a persistent setting value by key."""
        row = self.db.fetchone("SELECT value FROM settings WHERE key = ?;", (key,))
        if row:
            return str(row["value"])
        return default

    def set_setting(self, key: str, value: Any) -> None:
        """Store or update a setting key-value pair."""
        val_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        now = time.time()
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at;
                """,
                (key, val_str, now),
            )

    def get_all_settings(self) -> Dict[str, str]:
        """Fetch all stored key-value settings."""
        rows = self.db.fetchall("SELECT key, value FROM settings ORDER BY key;")
        return {r["key"]: r["value"] for r in rows}

    # =========================================================================
    # 5. Logs CRUD
    # =========================================================================

    def log_event(
        self,
        level: str,
        message: str,
        logger_name: str = "clashvision",
        timestamp: Optional[float] = None,
    ) -> int:
        """Record a structured application log event."""
        ts = timestamp if timestamp is not None else time.time()
        level_str = _sanitize_str(level).upper()
        logger_str = _sanitize_str(logger_name)
        msg_str = _sanitize_str(message)
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO logs (timestamp, level, logger_name, message)
                VALUES (?, ?, ?, ?);
                """,
                (ts, level_str, logger_str, msg_str),
            )
            return int(cursor.lastrowid)

    def query_logs(
        self,
        limit: int = 100,
        level: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query structured log entries with optional level and text filtering."""
        sql = "SELECT log_id, timestamp, level, logger_name, message FROM logs WHERE 1=1"
        params: List[Any] = []

        if level and level.upper() != "ALL":
            sql += " AND level = ?"
            params.append(level.upper())

        if search:
            sql += " AND message LIKE ?"
            params.append(f"%{search}%")

        sql += " ORDER BY timestamp DESC LIMIT ?;"
        params.append(limit)

        rows = self.db.fetchall(sql, tuple(params))
        return [
            {
                "log_id": r["log_id"],
                "timestamp": float(r["timestamp"]),
                "level": r["level"],
                "logger_name": r["logger_name"],
                "message": r["message"],
            }
            for r in rows
        ]

    def clear_logs(self) -> int:
        """Delete all log entries, returning deleted count."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM logs;")
            return cursor.rowcount

    # =========================================================================
    # 6. Dataset Annotations CRUD
    # =========================================================================

    def save_annotation(self, annotation: DatasetAnnotation) -> None:
        """Persist or update a dataset annotation item."""
        state_val = (
            annotation.state.value
            if isinstance(annotation.state, GameState)
            else str(annotation.state)
        )
        labels_str = json.dumps(annotation.labels or {})
        bboxes_str = json.dumps(annotation.bounding_boxes or [])
        now = time.time()

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO dataset_annotations (
                    annotation_id, image_path, timestamp, state, labels_json, bounding_boxes_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(annotation_id) DO UPDATE SET
                    image_path = excluded.image_path,
                    timestamp = excluded.timestamp,
                    state = excluded.state,
                    labels_json = excluded.labels_json,
                    bounding_boxes_json = excluded.bounding_boxes_json;
                """,
                (
                    annotation.annotation_id,
                    annotation.image_path,
                    now,
                    state_val,
                    labels_str,
                    bboxes_str,
                ),
            )

    def list_annotations(
        self,
        limit: int = 100,
        state: Optional[str] = None,
    ) -> List[DatasetAnnotation]:
        """List dataset annotations, optionally filtered by GameState."""
        if state and state.upper() != "ALL":
            rows = self.db.fetchall(
                """
                SELECT annotation_id, image_path, timestamp, state, labels_json, bounding_boxes_json
                FROM dataset_annotations
                WHERE state = ?
                ORDER BY timestamp DESC
                LIMIT ?;
                """,
                (state.upper(), limit),
            )
        else:
            rows = self.db.fetchall(
                """
                SELECT annotation_id, image_path, timestamp, state, labels_json, bounding_boxes_json
                FROM dataset_annotations
                ORDER BY timestamp DESC
                LIMIT ?;
                """,
                (limit,),
            )

        annotations = []
        for r in rows:
            try:
                st_enum = GameState(r["state"])
            except Exception:
                st_enum = GameState.UNKNOWN

            labels = {}
            if r["labels_json"]:
                try:
                    labels = json.loads(r["labels_json"])
                except Exception:
                    labels = {}

            bboxes = []
            if r["bounding_boxes_json"]:
                try:
                    bboxes = json.loads(r["bounding_boxes_json"])
                except Exception:
                    bboxes = []

            annotations.append(
                DatasetAnnotation(
                    annotation_id=r["annotation_id"],
                    image_path=r["image_path"],
                    state=st_enum,
                    labels=labels,
                    bounding_boxes=bboxes,
                )
            )
        return annotations

    def export_dataset_json(
        self,
        output_path: Optional[str] = None,
        state: Optional[str] = None,
    ) -> str:
        """Export dataset annotations as formatted JSON, optionally writing to disk."""
        annotations = self.list_annotations(limit=10000, state=state)
        data = [a.to_dict() for a in annotations]
        json_output = json.dumps(data, indent=2)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_output)

        return json_output
