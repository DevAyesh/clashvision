"""
Unit tests for ClashVision Storage Subsystem.
Tests SQLite WAL connection manager, schema initialization, and Repository CRUD operations.
"""

import json
import os
import tempfile
import threading
import time
from typing import Generator
import pytest

from clashvision.core.constants import GameState, StrategyType, RiskLevel
from clashvision.core.models import (
    SessionRecord,
    DatasetAnnotation,
    FarmingRecommendation,
    StrategyEvaluation,
    DeploymentStep,
)
from clashvision.storage.db import DatabaseManager, EXPECTED_TABLES
from clashvision.storage.repository import Repository


@pytest.fixture
def temp_db_file() -> Generator[str, None, None]:
    """Provides a path to a temporary database file and cleans it up."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    if os.path.exists(path):
        os.remove(path)
    yield path
    for p in [path, f"{path}-wal", f"{path}-shm"]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


@pytest.fixture
def file_repo(temp_db_file: str) -> Generator[Repository, None, None]:
    """File-backed Repository instance."""
    db = DatabaseManager(db_path=temp_db_file, auto_init=True)
    repo = Repository(db=db)
    yield repo
    db.close()


@pytest.fixture
def memory_repo() -> Generator[Repository, None, None]:
    """In-memory Repository instance."""
    db = DatabaseManager(db_path=":memory:", auto_init=True)
    repo = Repository(db=db)
    yield repo
    db.close()


class TestDatabaseManager:
    """Tests for DatabaseManager connection handling, WAL mode, and schema."""

    def test_schema_initialization_all_tables_created(self, temp_db_file: str) -> None:
        db = DatabaseManager(db_path=temp_db_file, auto_init=True)
        tables = db.get_table_names()
        for tbl in EXPECTED_TABLES:
            assert tbl in tables, f"Table {tbl} was not created"
        assert db.verify_schema_integrity() is True
        db.close()

    def test_wal_mode_enabled_on_disk(self, temp_db_file: str) -> None:
        db = DatabaseManager(db_path=temp_db_file, auto_init=True)
        mode = db.check_wal_mode()
        assert mode == "wal"
        db.close()

    def test_foreign_keys_pragmas_enforced(self, temp_db_file: str) -> None:
        db = DatabaseManager(db_path=temp_db_file, auto_init=True)
        row = db.fetchone("PRAGMA foreign_keys;")
        assert row is not None
        assert row[0] == 1
        db.close()

    def test_table_exists_method(self, memory_repo: Repository) -> None:
        db = memory_repo.db
        assert db.table_exists("sessions") is True
        assert db.table_exists("base_scans") is True
        assert db.table_exists("non_existent_table") is False

    def test_concurrent_threads_reading_writing(self, temp_db_file: str) -> None:
        db = DatabaseManager(db_path=temp_db_file, auto_init=True)
        repo = Repository(db=db)

        def worker(thread_idx: int) -> None:
            for i in range(10):
                repo.log_event("INFO", f"Thread {thread_idx} log entry {i}", logger_name=f"worker_{thread_idx}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        logs = repo.query_logs(limit=100)
        assert len(logs) == 50
        db.close()


class TestRepositorySessions:
    """Tests for session management CRUD."""

    def test_create_and_get_session(self, memory_repo: Repository) -> None:
        sess = memory_repo.create_session("sess_alpha", start_time=1000.0, notes="Test session")
        assert sess.session_id == "sess_alpha"
        assert sess.start_time == 1000.0
        assert sess.notes == "Test session"
        assert sess.end_time is None

        fetched = memory_repo.get_session("sess_alpha")
        assert fetched is not None
        assert fetched.session_id == "sess_alpha"
        assert fetched.start_time == 1000.0
        assert fetched.notes == "Test session"

    def test_end_session(self, memory_repo: Repository) -> None:
        memory_repo.create_session("sess_beta", start_time=100.0)
        ended = memory_repo.end_session("sess_beta", end_time=250.0, notes="Finished cleanly")
        assert ended is not None
        assert ended.end_time == 250.0
        assert ended.notes == "Finished cleanly"

    def test_list_sessions_order_and_pagination(self, memory_repo: Repository) -> None:
        for i in range(5):
            memory_repo.create_session(f"sess_{i}", start_time=100.0 + i)

        sessions = memory_repo.list_sessions(limit=3, offset=0)
        assert len(sessions) == 3
        # Should be ordered descending by start_time
        assert sessions[0].session_id == "sess_4"
        assert sessions[1].session_id == "sess_3"
        assert sessions[2].session_id == "sess_2"

    def test_increment_session_scans(self, memory_repo: Repository) -> None:
        memory_repo.create_session("sess_counter", start_time=50.0)
        memory_repo.increment_session_scans("sess_counter", matched=False)
        memory_repo.increment_session_scans("sess_counter", matched=True)
        memory_repo.increment_session_scans("sess_counter", matched=True)

        s = memory_repo.get_session("sess_counter")
        assert s is not None
        assert s.total_scans == 3
        assert s.matched_targets == 2


class TestRepositoryBaseScans:
    """Tests for base scan records."""

    def test_save_and_retrieve_base_scan(self, memory_repo: Repository) -> None:
        memory_repo.create_session("sess_scan_1", start_time=500.0)
        scan_id = memory_repo.save_base_scan(
            session_id="sess_scan_1",
            timestamp=505.0,
            game_state=GameState.ENEMY_BASE_PREVIEW,
            confidence=0.98,
            gold=650000,
            elixir=700000,
            dark_elixir=4500,
            th_accessible=True,
            raw_ocr_json={"gold_raw": "650,000"},
        )
        assert scan_id > 0

        scans = memory_repo.get_base_scans(session_id="sess_scan_1")
        assert len(scans) == 1
        s0 = scans[0]
        assert s0["scan_id"] == scan_id
        assert s0["gold"] == 650000
        assert s0["elixir"] == 700000
        assert s0["dark_elixir"] == 4500
        assert s0["th_accessible"] is True
        assert s0["game_state"] == "ENEMY_BASE_PREVIEW"

        count = memory_repo.get_scan_count("sess_scan_1")
        assert count == 1


class TestRepositoryDecisionsAndEvaluations:
    """Tests for decisions and child strategy evaluations."""

    def test_save_recommendation_and_query(self, memory_repo: Repository) -> None:
        memory_repo.create_session("sess_rec", start_time=100.0)
        scan_id = memory_repo.save_base_scan(
            session_id="sess_rec",
            timestamp=105.0,
            game_state=GameState.ENEMY_BASE_PREVIEW,
            gold=800000,
            elixir=800000,
        )

        rec = FarmingRecommendation(
            suitability_score=85.5,
            primary_strategy=StrategyType.SNEAKY_GOBLIN,
            rankings=[
                StrategyEvaluation(
                    strategy_type=StrategyType.SNEAKY_GOBLIN,
                    score=85.5,
                    viable=True,
                    risk_level=RiskLevel.LOW,
                    reasons=["High collector loot", "Exposed storages"],
                ),
                StrategyEvaluation(
                    strategy_type=StrategyType.BARCH,
                    score=60.0,
                    viable=True,
                    risk_level=RiskLevel.MEDIUM,
                    reasons=["Good perimeter"],
                ),
            ],
            deployment_plan=[
                DeploymentStep(step_id=1, troop_type="Sneaky Goblin", target_x=100, target_y=200, delay_ms=500),
            ],
            score_breakdown={"resource_score": 90.0, "accessibility_score": 80.0},
            timestamp=106.0,
            explanation="Excellent sneaky goblin target with exposed resource nodes.",
        )

        dec_id = memory_repo.save_recommendation(
            rec,
            session_id="sess_rec",
            scan_id=scan_id,
        )
        assert dec_id > 0

        decisions = memory_repo.get_recent_decisions(session_id="sess_rec", limit=10)
        assert len(decisions) == 1
        d = decisions[0]
        assert d["decision_id"] == dec_id
        assert d["primary_strategy"] == "SNEAKY_GOBLIN"
        assert d["suitability_score"] == 85.5
        assert len(d["evaluations"]) == 2
        assert d["evaluations"][0]["strategy_type"] == "SNEAKY_GOBLIN"
        assert d["evaluations"][0]["reasons"] == ["High collector loot", "Exposed storages"]


class TestRepositorySettings:
    """Tests for key-value configuration storage."""

    def test_set_and_get_setting(self, memory_repo: Repository) -> None:
        memory_repo.set_setting("target_fps", 30)
        memory_repo.set_setting("active_engine", "pytesseract")

        assert memory_repo.get_setting("target_fps") == "30"
        assert memory_repo.get_setting("active_engine") == "pytesseract"
        assert memory_repo.get_setting("missing_key", default="default_val") == "default_val"

    def test_upsert_setting(self, memory_repo: Repository) -> None:
        memory_repo.set_setting("min_gold", 400000)
        assert memory_repo.get_setting("min_gold") == "400000"
        memory_repo.set_setting("min_gold", 600000)
        assert memory_repo.get_setting("min_gold") == "600000"

    def test_get_all_settings(self, memory_repo: Repository) -> None:
        memory_repo.set_setting("opt_a", "1")
        memory_repo.set_setting("opt_b", "2")
        all_s = memory_repo.get_all_settings()
        assert all_s["opt_a"] == "1"
        assert all_s["opt_b"] == "2"


class TestRepositoryLogs:
    """Tests for structured logging table."""

    def test_log_event_and_filtering(self, memory_repo: Repository) -> None:
        memory_repo.log_event("INFO", "Starting application", logger_name="main")
        memory_repo.log_event("WARNING", "Capture frame dropped", logger_name="capture")
        memory_repo.log_event("ERROR", "Window handle invalid", logger_name="capture")

        all_logs = memory_repo.query_logs(limit=10)
        assert len(all_logs) == 3

        error_logs = memory_repo.query_logs(level="ERROR")
        assert len(error_logs) == 1
        assert error_logs[0]["message"] == "Window handle invalid"

        search_logs = memory_repo.query_logs(search="frame dropped")
        assert len(search_logs) == 1
        assert search_logs[0]["level"] == "WARNING"

    def test_clear_logs(self, memory_repo: Repository) -> None:
        memory_repo.log_event("INFO", "msg 1")
        memory_repo.log_event("INFO", "msg 2")
        deleted = memory_repo.clear_logs()
        assert deleted == 2
        assert len(memory_repo.query_logs()) == 0


class TestRepositoryDatasetAnnotations:
    """Tests for dataset annotation management and JSON export."""

    def test_save_and_list_annotations(self, memory_repo: Repository) -> None:
        ann = DatasetAnnotation(
            annotation_id="ann_001",
            image_path="/images/sample1.png",
            state=GameState.ENEMY_BASE_PREVIEW,
            labels={"th_level": 14},
            bounding_boxes=[{"label": "town_hall", "x": 100, "y": 120, "w": 50, "h": 50}],
        )
        memory_repo.save_annotation(ann)

        listed = memory_repo.list_annotations()
        assert len(listed) == 1
        a = listed[0]
        assert a.annotation_id == "ann_001"
        assert a.state == GameState.ENEMY_BASE_PREVIEW
        assert a.labels == {"th_level": 14}
        assert len(a.bounding_boxes) == 1

    def test_export_dataset_json(self, memory_repo: Repository, tmp_path) -> None:
        ann = DatasetAnnotation(
            annotation_id="ann_export",
            image_path="/images/sample2.png",
            state=GameState.HOME_VILLAGE,
            labels={"notes": "clean view"},
            bounding_boxes=[],
        )
        memory_repo.save_annotation(ann)

        out_file = str(tmp_path / "dataset_export.json")
        json_str = memory_repo.export_dataset_json(output_path=out_file)
        assert os.path.exists(out_file)
        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["annotation_id"] == "ann_export"
        assert data[0]["state"] == "HOME_VILLAGE"
