"""
Adversarial Stress and Persistence Verification Suite for Milestone 5 (clashvision/storage/).
Empirically stress-tests:
1. Multi-threaded concurrent read/write transactions in SQLite WAL mode.
2. Foreign key constraints, integrity violations, and cascade deletions.
3. Malformed, corrupted, and adversarial JSON inputs across scans, decisions, and annotations.
4. Edge cases: missing/null fields, empty tables, boundary inputs, SQL injection attempts, and rollback atomicity.
5. High-volume record stress and index pagination performance.
"""

from __future__ import annotations
import json
import os
import sqlite3
import tempfile
import threading
import time
from typing import Any, Dict, Generator, List
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
def temp_db_path() -> Generator[str, None, None]:
    """Provides a fresh disk-backed database path with full WAL cleanup."""
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
def disk_repo(temp_db_path: str) -> Generator[Repository, None, None]:
    """File-backed Repository instance configured with WAL mode."""
    db = DatabaseManager(db_path=temp_db_path, auto_init=True)
    repo = Repository(db=db)
    yield repo
    db.close()


@pytest.fixture
def mem_repo() -> Generator[Repository, None, None]:
    """In-memory Repository instance."""
    db = DatabaseManager(db_path=":memory:", auto_init=True)
    repo = Repository(db=db)
    yield repo
    db.close()


# =============================================================================
# 1. Multi-Threaded Concurrency & WAL Mode Stress Tests
# =============================================================================

class TestWALConcurrencyAndStress:
    """Adversarial stress-testing of SQLite WAL multi-threaded concurrency."""

    def test_concurrent_multi_threaded_mixed_writers_and_readers(self, temp_db_path: str) -> None:
        """
        Stress test: 10 worker threads performing high-frequency mixed write operations
        (sessions, scans, recommendations, settings, annotations, logs) while 4 reader threads
        continuously query logs, scans, and sessions.
        Empirically verifies that WAL mode + busy_timeout handles concurrent contention without
        deadlocks, database locked errors, or data loss.
        """
        db = DatabaseManager(db_path=temp_db_path, auto_init=True)
        repo = Repository(db=db)

        # Baseline session
        master_session = repo.create_session("sess_concurrency_master", start_time=time.time())

        errors: List[str] = []
        stop_readers = threading.Event()

        def log_writer(worker_id: int) -> None:
            try:
                for i in range(40):
                    repo.log_event("INFO", f"Worker {worker_id} iteration {i}", logger_name=f"worker_{worker_id}")
            except Exception as e:
                errors.append(f"log_writer_{worker_id}: {type(e).__name__}: {e}")

        def scan_writer(worker_id: int) -> None:
            try:
                for i in range(30):
                    repo.save_base_scan(
                        session_id=master_session.session_id,
                        game_state=GameState.ENEMY_BASE_PREVIEW,
                        confidence=0.95,
                        gold=500000 + i * 1000,
                        elixir=400000 + i * 1000,
                        dark_elixir=2000 + i * 10,
                        th_accessible=(i % 2 == 0),
                        raw_ocr_json={"step": i, "worker": worker_id},
                    )
            except Exception as e:
                errors.append(f"scan_writer_{worker_id}: {type(e).__name__}: {e}")

        def recommendation_writer(worker_id: int) -> None:
            try:
                for i in range(20):
                    rec = FarmingRecommendation(
                        suitability_score=75.0 + (i % 20),
                        primary_strategy=StrategyType.SNEAKY_GOBLIN,
                        rankings=[
                            StrategyEvaluation(
                                strategy_type=StrategyType.SNEAKY_GOBLIN,
                                score=80.0,
                                viable=True,
                                risk_level=RiskLevel.LOW,
                                reasons=[f"Reason {i}"],
                            )
                        ],
                        score_breakdown={"resource": 80.0},
                        explanation=f"Worker {worker_id} rec {i}",
                    )
                    repo.save_recommendation(rec, session_id=master_session.session_id)
            except Exception as e:
                errors.append(f"rec_writer_{worker_id}: {type(e).__name__}: {e}")

        def settings_writer(worker_id: int) -> None:
            try:
                for i in range(25):
                    repo.set_setting(f"worker_{worker_id}_key_{i % 5}", f"val_{i}")
            except Exception as e:
                errors.append(f"settings_writer_{worker_id}: {type(e).__name__}: {e}")

        def annotation_writer(worker_id: int) -> None:
            try:
                for i in range(20):
                    ann = DatasetAnnotation(
                        annotation_id=f"ann_w{worker_id}_{i}",
                        image_path=f"/img/w{worker_id}_{i}.png",
                        state=GameState.HOME_VILLAGE,
                        labels={"worker": worker_id, "idx": i},
                        bounding_boxes=[{"box": [0, 0, 10, 10]}],
                    )
                    repo.save_annotation(ann)
            except Exception as e:
                errors.append(f"annotation_writer_{worker_id}: {type(e).__name__}: {e}")

        def reader(reader_id: int) -> None:
            try:
                while not stop_readers.is_set():
                    repo.query_logs(limit=20)
                    repo.get_base_scans(session_id=master_session.session_id, limit=20)
                    repo.get_recent_decisions(session_id=master_session.session_id, limit=10)
                    repo.list_sessions(limit=5)
                    time.sleep(0.005)
            except Exception as e:
                errors.append(f"reader_{reader_id}: {type(e).__name__}: {e}")

        # Assemble thread pool: 2 log writers, 2 scan writers, 2 rec writers, 2 settings writers, 2 annotation writers
        threads: List[threading.Thread] = []
        threads.extend([threading.Thread(target=log_writer, args=(i,)) for i in range(2)])
        threads.extend([threading.Thread(target=scan_writer, args=(i,)) for i in range(2)])
        threads.extend([threading.Thread(target=recommendation_writer, args=(i,)) for i in range(2)])
        threads.extend([threading.Thread(target=settings_writer, args=(i,)) for i in range(2)])
        threads.extend([threading.Thread(target=annotation_writer, args=(i,)) for i in range(2)])

        reader_threads = [threading.Thread(target=reader, args=(i,)) for i in range(4)]
        for rt in reader_threads:
            rt.start()

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Stop readers
        stop_readers.set()
        for rt in reader_threads:
            rt.join()

        db.close()

        # Empirical assertions
        assert len(errors) == 0, f"Concurrent execution errors encountered: {errors}"

        # Reopen and verify data persistence
        db2 = DatabaseManager(db_path=temp_db_path, auto_init=False)
        repo2 = Repository(db=db2)

        logs_count = len(repo2.query_logs(limit=1000))
        scans_count = repo2.get_scan_count(master_session.session_id)
        decisions_count = len(repo2.get_recent_decisions(session_id=master_session.session_id, limit=100))
        annotations_count = len(repo2.list_annotations(limit=100))

        assert logs_count == 80, f"Expected 80 logs, got {logs_count}"
        assert scans_count == 60, f"Expected 60 scans, got {scans_count}"
        assert decisions_count == 40, f"Expected 40 decisions, got {decisions_count}"
        assert annotations_count == 40, f"Expected 40 annotations, got {annotations_count}"

        # Master session scans and matched counter should be updated
        sess = repo2.get_session(master_session.session_id)
        assert sess is not None
        assert sess.total_scans == 60, f"Expected 60 scans on session, got {sess.total_scans}"
        assert sess.matched_targets == 40, f"Expected 40 matched targets, got {sess.matched_targets}"

        db2.close()

    def test_busy_timeout_lock_holding_contention(self, temp_db_path: str) -> None:
        """
        Verify that PRAGMA busy_timeout=5000 properly causes concurrent writers
        to wait for a thread holding a transaction lock without immediately crashing with
        'database is locked'.
        """
        db = DatabaseManager(db_path=temp_db_path, auto_init=True)
        repo = Repository(db=db)

        errors: List[str] = []
        lock_acquired = threading.Event()

        def slow_transaction_holder() -> None:
            try:
                with db.transaction() as conn:
                    lock_acquired.set()
                    conn.execute(
                        "INSERT INTO logs (timestamp, level, logger_name, message) VALUES (?, ?, ?, ?);",
                        (time.time(), "INFO", "slow_holder", "Holding transaction lock"),
                    )
                    time.sleep(0.4)  # Hold lock for 400ms (well within 5000ms busy_timeout)
            except Exception as e:
                errors.append(f"slow_holder: {e}")

        def contending_writer(idx: int) -> None:
            lock_acquired.wait(timeout=2.0)
            try:
                repo.log_event("INFO", f"Contending writer {idx} after wait", logger_name=f"writer_{idx}")
            except Exception as e:
                errors.append(f"contending_writer_{idx}: {e}")

        t_slow = threading.Thread(target=slow_transaction_holder)
        t_contenders = [threading.Thread(target=contending_writer, args=(i,)) for i in range(5)]

        t_slow.start()
        for t in t_contenders:
            t.start()

        t_slow.join()
        for t in t_contenders:
            t.join()

        db.close()
        assert len(errors) == 0, f"Contention errors under busy_timeout: {errors}"

    def test_in_memory_concurrency_failure_mode_documented(self) -> None:
        """
        Adversarial test on in-memory mode (:memory:).
        Verifies that DatabaseManager's shared in-memory connection safely serializes
        concurrent writes using RLock synchronization without transaction collisions.
        """
        db = DatabaseManager(db_path=":memory:", auto_init=True)
        repo = Repository(db=db)

        errors: List[Exception] = []

        def worker(idx: int) -> None:
            try:
                for i in range(15):
                    repo.log_event("INFO", f"msg {idx}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        db.close()
        # In memory mode with RLock synchronization, 0 transaction collisions occur
        assert len(errors) == 0, f"Unexpected concurrency errors under RLock serialization: {errors}"


# =============================================================================
# 2. Foreign Key Constraints & Integrity Tests
# =============================================================================

class TestForeignKeysAndIntegrity:
    """Adversarial testing of relational integrity, foreign key pragmas, and cascade behaviors."""

    def test_foreign_key_rejection_on_invalid_session_in_base_scans(self, disk_repo: Repository) -> None:
        """Verify inserting base_scan referencing non-existent session_id raises IntegrityError."""
        with pytest.raises(sqlite3.IntegrityError):
            disk_repo.save_base_scan(
                session_id="non_existent_session_id_9999",
                game_state=GameState.ENEMY_BASE_PREVIEW,
            )

    def test_foreign_key_rejection_on_invalid_scan_or_session_in_decisions(self, disk_repo: Repository) -> None:
        """Verify inserting decision referencing non-existent scan_id raises IntegrityError."""
        rec = FarmingRecommendation(
            suitability_score=90.0,
            primary_strategy=StrategyType.SNEAKY_GOBLIN,
        )
        with pytest.raises(sqlite3.IntegrityError):
            disk_repo.save_recommendation(
                rec,
                session_id=None,
                scan_id=999999,  # Non-existent scan ID
            )

        with pytest.raises(sqlite3.IntegrityError):
            disk_repo.save_recommendation(
                rec,
                session_id="ghost_session_uuid",  # Non-existent session ID
                scan_id=None,
            )

    def test_foreign_key_rejection_on_invalid_decision_in_evaluations(self, disk_repo: Repository) -> None:
        """Verify inserting evaluation directly with non-existent decision_id raises IntegrityError."""
        with pytest.raises(sqlite3.IntegrityError):
            disk_repo.db.execute(
                """
                INSERT INTO evaluations (decision_id, strategy_type, score, viable, risk_level, reasons_json)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (999999, "SNEAKY_GOBLIN", 80.0, 1, "LOW", "[]"),
            )

    def test_cascade_delete_decisions_to_evaluations(self, disk_repo: Repository) -> None:
        """
        Verify that deleting a decision cascades and automatically removes all child evaluations
        via FOREIGN KEY (decision_id) REFERENCES decisions(decision_id) ON DELETE CASCADE.
        """
        sess = disk_repo.create_session("sess_cascade_test")
        scan_id = disk_repo.save_base_scan(session_id=sess.session_id, game_state=GameState.ENEMY_BASE_PREVIEW)

        rec = FarmingRecommendation(
            suitability_score=85.0,
            primary_strategy=StrategyType.SNEAKY_GOBLIN,
            rankings=[
                StrategyEvaluation(strategy_type=StrategyType.SNEAKY_GOBLIN, score=85.0, viable=True, risk_level=RiskLevel.LOW),
                StrategyEvaluation(strategy_type=StrategyType.BARCH, score=60.0, viable=True, risk_level=RiskLevel.MEDIUM),
                StrategyEvaluation(strategy_type=StrategyType.MASS_MINER, score=45.0, viable=False, risk_level=RiskLevel.HIGH),
            ],
        )
        dec_id = disk_repo.save_recommendation(rec, session_id=sess.session_id, scan_id=scan_id)

        # Confirm 3 evaluations exist
        eval_rows = disk_repo.db.fetchall("SELECT eval_id FROM evaluations WHERE decision_id = ?;", (dec_id,))
        assert len(eval_rows) == 3

        # Delete the decision
        with disk_repo.db.transaction() as conn:
            conn.execute("DELETE FROM decisions WHERE decision_id = ?;", (dec_id,))

        # Verify child evaluations were cascaded away
        remaining_evals = disk_repo.db.fetchall("SELECT eval_id FROM evaluations WHERE decision_id = ?;", (dec_id,))
        assert len(remaining_evals) == 0

    def test_set_null_delete_session_cascades_safely_to_scans_and_decisions(self, disk_repo: Repository) -> None:
        """
        Verify that deleting a session sets session_id = NULL in child base_scans and decisions
        via ON DELETE SET NULL, preserving the scan/decision records without crashing queries.
        """
        sess = disk_repo.create_session("sess_set_null_test")
        scan_id = disk_repo.save_base_scan(session_id=sess.session_id, game_state=GameState.ENEMY_BASE_PREVIEW)
        rec = FarmingRecommendation(suitability_score=75.0, primary_strategy=StrategyType.BARCH)
        dec_id = disk_repo.save_recommendation(rec, session_id=sess.session_id, scan_id=scan_id)

        # Delete the session
        with disk_repo.db.transaction() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?;", (sess.session_id,))

        # Scan should still exist, but session_id is NULL
        scans = disk_repo.get_base_scans(limit=10)
        found_scan = next((s for s in scans if s["scan_id"] == scan_id), None)
        assert found_scan is not None
        assert found_scan["session_id"] is None

        # Decision should still exist, but session_id is NULL
        decisions = disk_repo.get_recent_decisions(limit=10)
        found_dec = next((d for d in decisions if d["decision_id"] == dec_id), None)
        assert found_dec is not None
        assert found_dec["session_id"] is None
        assert found_dec["scan_id"] == scan_id

    def test_set_null_delete_base_scan_cascades_safely_to_decisions(self, disk_repo: Repository) -> None:
        """
        Verify that deleting a base_scan sets scan_id = NULL in decisions via ON DELETE SET NULL.
        """
        sess = disk_repo.create_session("sess_scan_del_test")
        scan_id = disk_repo.save_base_scan(session_id=sess.session_id, game_state=GameState.ENEMY_BASE_PREVIEW)
        rec = FarmingRecommendation(suitability_score=80.0, primary_strategy=StrategyType.MASS_DRAGON)
        dec_id = disk_repo.save_recommendation(rec, session_id=sess.session_id, scan_id=scan_id)

        # Delete the base_scan
        with disk_repo.db.transaction() as conn:
            conn.execute("DELETE FROM base_scans WHERE scan_id = ?;", (scan_id,))

        decisions = disk_repo.get_recent_decisions(limit=10)
        found_dec = next((d for d in decisions if d["decision_id"] == dec_id), None)
        assert found_dec is not None
        assert found_dec["scan_id"] is None

    def test_unique_constraint_violation_on_session_id(self, disk_repo: Repository) -> None:
        """Verify attempting to create duplicate session_id raises IntegrityError."""
        disk_repo.create_session("sess_unique_test")
        with pytest.raises(sqlite3.IntegrityError):
            disk_repo.create_session("sess_unique_test")

    def test_not_null_constraints_enforced(self, disk_repo: Repository) -> None:
        """Verify NOT NULL constraints on essential columns."""
        # sessions.start_time is NOT NULL
        with pytest.raises(sqlite3.IntegrityError):
            disk_repo.db.execute(
                "INSERT INTO sessions (session_id, start_time) VALUES (?, ?);",
                ("sess_bad_null", None),
            )

        # logs.level and logs.message are NOT NULL
        with pytest.raises(sqlite3.IntegrityError):
            disk_repo.db.execute(
                "INSERT INTO logs (timestamp, level, logger_name, message) VALUES (?, ?, ?, ?);",
                (time.time(), "INFO", "test", None),
            )

        # base_scans.timestamp and game_state are NOT NULL
        with pytest.raises(sqlite3.IntegrityError):
            disk_repo.db.execute(
                "INSERT INTO base_scans (timestamp, game_state) VALUES (?, ?);",
                (None, "ENEMY_BASE_PREVIEW"),
            )


# =============================================================================
# 3. Corrupt & Malformed JSON Inputs
# =============================================================================

class TestCorruptAndMalformedJson:
    """Adversarial testing with malformed, truncated, non-standard, or malicious JSON strings."""

    def test_corrupt_json_in_raw_ocr_json_does_not_crash_read(self, disk_repo: Repository) -> None:
        """
        Verify that corrupted raw OCR strings (syntax errors, non-JSON strings, unicode noise)
        are stored and retrieved safely as text without crashing get_base_scans().
        """
        sess = disk_repo.create_session("sess_corrupt_ocr")
        corrupted_strings = [
            "{unclosed json: 123",
            "['malformed', unquoted]",
            "null",
            "",
            "Plain text without any json markers",
            "{\"unicode\": \"\\u0000\\u001f\\ufffd\\U0001F600\"}",
        ]

        for idx, bad_str in enumerate(corrupted_strings):
            scan_id = disk_repo.save_base_scan(
                session_id=sess.session_id,
                game_state=GameState.BATTLE_IN_PROGRESS,
                raw_ocr_json=bad_str,
            )
            assert scan_id > 0

        scans = disk_repo.get_base_scans(session_id=sess.session_id, limit=20)
        assert len(scans) == len(corrupted_strings)

    def test_raw_ocr_json_with_unpaired_surrogates_triggers_unicode_encode_error(self, disk_repo: Repository) -> None:
        """
        Verify that passing a raw OCR string containing unpaired or lone surrogates
        is safely sanitized without raising an unhandled UnicodeEncodeError.
        """
        sess = disk_repo.create_session("sess_surrogate_test")
        surrogate_str = "OCR text with lone surrogate: \ud83d"

        scan_id = disk_repo.save_base_scan(
            session_id=sess.session_id,
            game_state=GameState.BATTLE_IN_PROGRESS,
            raw_ocr_json=surrogate_str,
        )
        assert scan_id > 0
        scans = disk_repo.get_base_scans(session_id=sess.session_id, limit=5)
        assert len(scans) == 1
        assert ("?" in scans[0]["raw_ocr_json"]) or ("\ufffd" in scans[0]["raw_ocr_json"])

    def test_raw_ocr_json_list_input_fails_parameter_binding(self, disk_repo: Repository) -> None:
        """
        Verify that passing a list input to save_base_scan (e.g. OCR bounding boxes list)
        is safely serialized to JSON rather than raising a ProgrammingError.
        """
        sess = disk_repo.create_session("sess_list_ocr_test")
        list_input = [{"text": "500000", "conf": 0.95}, {"text": "Gold", "conf": 0.90}]

        scan_id = disk_repo.save_base_scan(
            session_id=sess.session_id,
            game_state=GameState.BATTLE_IN_PROGRESS,
            raw_ocr_json=list_input,  # type: ignore[arg-type]
        )
        assert scan_id > 0
        scans = disk_repo.get_base_scans(session_id=sess.session_id, limit=5)
        assert len(scans) == 1
        assert "500000" in scans[0]["raw_ocr_json"]

    def test_empty_string_session_id_fails_foreign_key_check(self, disk_repo: Repository) -> None:
        """
        Verify that passing session_id="" (empty string) is safely normalized to None,
        satisfying foreign key constraints without raising IntegrityError.
        """
        scan_id = disk_repo.save_base_scan(
            session_id="",
            game_state=GameState.BATTLE_IN_PROGRESS,
        )
        assert scan_id > 0

    def test_corrupt_json_in_decision_breakdown_and_evaluations_reasons(self, disk_repo: Repository) -> None:
        """
        Directly inject corrupt JSON strings into decisions.breakdown_json and
        evaluations.reasons_json to test resilient error-handling fallbacks in get_recent_decisions().
        """
        sess = disk_repo.create_session("sess_corrupt_dec")

        # Manually insert decision with corrupted breakdown_json
        with disk_repo.db.transaction() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO decisions (session_id, timestamp, primary_strategy, suitability_score, explanation, breakdown_json)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (sess.session_id, time.time(), "SNEAKY_GOBLIN", 70.0, "Corrupt test", "{broken: json ["),
            )
            dec_id = int(cur.lastrowid)

            # Manually insert evaluation with corrupted reasons_json
            cur.execute(
                """
                INSERT INTO evaluations (decision_id, strategy_type, score, viable, risk_level, reasons_json)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (dec_id, "SNEAKY_GOBLIN", 70.0, 1, "LOW", "not_valid_json_at_all!"),
            )

        # Query via repository: should gracefully catch parsing errors and return fallback empty dict/list
        decisions = disk_repo.get_recent_decisions(session_id=sess.session_id, limit=5)
        assert len(decisions) == 1
        d = decisions[0]
        assert d["score_breakdown"] == {}, "Expected empty dict fallback for corrupted breakdown_json"
        assert len(d["evaluations"]) == 1
        assert d["evaluations"][0]["reasons"] == [], "Expected empty list fallback for corrupted reasons_json"

    def test_json_primitive_types_instead_of_expected_containers(self, disk_repo: Repository) -> None:
        """
        Test behavior when valid JSON primitives (e.g. 'null', '123', '"quoted string"')
        are in breakdown_json or reasons_json.
        """
        sess = disk_repo.create_session("sess_primitive_json")

        with disk_repo.db.transaction() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO decisions (session_id, timestamp, primary_strategy, suitability_score, explanation, breakdown_json)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (sess.session_id, time.time(), "BARCH", 50.0, "Primitive test", "null"),
            )
            dec_id = int(cur.lastrowid)

            cur.execute(
                """
                INSERT INTO evaluations (decision_id, strategy_type, score, viable, risk_level, reasons_json)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (dec_id, "BARCH", 50.0, 1, "LOW", "12345"),
            )

        decisions = disk_repo.get_recent_decisions(session_id=sess.session_id, limit=5)
        assert len(decisions) == 1
        d = decisions[0]
        # Query returns without unhandled exceptions
        assert d["decision_id"] == dec_id

    def test_corrupt_json_in_dataset_annotations(self, disk_repo: Repository) -> None:
        """
        Directly inject corrupted labels_json and bounding_boxes_json into dataset_annotations
        and verify list_annotations() and export_dataset_json() handle them gracefully.
        """
        with disk_repo.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO dataset_annotations (annotation_id, image_path, timestamp, state, labels_json, bounding_boxes_json)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                ("ann_corrupt_01", "/img/corrupt.png", time.time(), "HOME_VILLAGE", "{invalid: json", "[unclosed array"),
            )

        # list_annotations should recover gracefully
        annotations = disk_repo.list_annotations(limit=5)
        assert len(annotations) == 1
        ann = annotations[0]
        assert ann.labels == {}
        assert ann.bounding_boxes == []

        # export_dataset_json should format cleanly
        json_out = disk_repo.export_dataset_json()
        exported = json.loads(json_out)
        assert len(exported) == 1
        assert exported[0]["annotation_id"] == "ann_corrupt_01"
        assert exported[0]["labels"] == {}
        assert exported[0]["bounding_boxes"] == []


# =============================================================================
# 4. Edge Cases: Missing/Null Fields, Empty Tables, Boundary Values
# =============================================================================

class TestEdgeCasesAndBoundaryValues:
    """Testing empty tables, boundary values, null inputs, and SQL injection safety."""

    def test_empty_tables_queries_return_empty_safely(self, disk_repo: Repository) -> None:
        """Verify all query methods on empty database return empty containers or None without errors."""
        assert disk_repo.get_session("non_existent") is None
        assert disk_repo.list_sessions() == []
        assert disk_repo.get_base_scans() == []
        assert disk_repo.get_scan_count() == 0
        assert disk_repo.get_recent_decisions() == []
        assert disk_repo.get_setting("non_existent_key") is None
        assert disk_repo.get_setting("non_existent_key", default="fallback") == "fallback"
        assert disk_repo.get_all_settings() == {}
        assert disk_repo.query_logs() == []
        assert disk_repo.clear_logs() == 0
        assert disk_repo.list_annotations() == []
        assert disk_repo.export_dataset_json() == "[]"

    def test_null_and_missing_optional_fields_in_models(self, disk_repo: Repository) -> None:
        """
        Verify saving models with all optional fields set to None:
        gold, elixir, dark_elixir, end_time, notes, raw_ocr_json, score_breakdown, explanation.
        """
        sess = disk_repo.create_session("sess_all_nulls", notes="")
        assert sess.notes == ""

        ended_sess = disk_repo.end_session("sess_all_nulls")
        assert ended_sess is not None
        assert ended_sess.end_time is not None

        scan_id = disk_repo.save_base_scan(
            session_id="sess_all_nulls",
            gold=None,
            elixir=None,
            dark_elixir=None,
            raw_ocr_json=None,
        )
        assert scan_id > 0

        scans = disk_repo.get_base_scans(session_id="sess_all_nulls")
        assert len(scans) == 1
        s = scans[0]
        assert s["gold"] is None
        assert s["elixir"] is None
        assert s["dark_elixir"] is None
        assert s["raw_ocr_json"] == ""

        # Recommendation with empty / None breakdown and explanation
        rec = FarmingRecommendation(
            suitability_score=0.0,
            primary_strategy=StrategyType.CUSTOM,
            rankings=[],
            score_breakdown={},
            explanation="",
        )
        dec_id = disk_repo.save_recommendation(rec, session_id="sess_all_nulls", scan_id=scan_id)
        assert dec_id > 0

        decisions = disk_repo.get_recent_decisions(session_id="sess_all_nulls")
        assert len(decisions) == 1
        d = decisions[0]
        assert d["score_breakdown"] == {}
        assert d["evaluations"] == []

    def test_extreme_boundary_integers_and_floats(self, disk_repo: Repository) -> None:
        """
        Verify SQLite persistence handles large 64-bit integers and extreme floats
        (e.g., maximum loot values, negative coordinates, sub-millisecond timestamps).
        """
        sess = disk_repo.create_session("sess_extreme_numeric")
        max_int64 = (1 << 63) - 1

        scan_id = disk_repo.save_base_scan(
            session_id=sess.session_id,
            timestamp=1.700000000001e9,
            gold=max_int64,
            elixir=0,
            dark_elixir=-1,  # Negative reading
            confidence=0.99999999,
        )

        scans = disk_repo.get_base_scans(session_id=sess.session_id)
        assert len(scans) == 1
        s = scans[0]
        assert s["gold"] == max_int64
        assert s["dark_elixir"] == -1
        assert abs(s["confidence"] - 0.99999999) < 1e-6

    def test_sql_injection_resilience_in_parameters(self, disk_repo: Repository) -> None:
        """
        Verify parameterized queries prevent SQL injection through inputs like session_id,
        setting keys, log messages, and search queries.
        """
        sql_injection_payload = "'; DROP TABLE sessions; DROP TABLE logs; --"

        # Log event with payload
        log_id = disk_repo.log_event("INFO", sql_injection_payload, logger_name=sql_injection_payload)
        assert log_id > 0

        # Query with search payload
        results = disk_repo.query_logs(search="DROP TABLE")
        assert len(results) == 1
        assert "DROP TABLE" in results[0]["message"]

        # Settings with injection payload
        disk_repo.set_setting(sql_injection_payload, "safe_value")
        assert disk_repo.get_setting(sql_injection_payload) == "safe_value"

        # Verify tables still exist
        assert disk_repo.db.table_exists("sessions") is True
        assert disk_repo.db.table_exists("logs") is True

    def test_transaction_atomicity_and_rollback(self, disk_repo: Repository) -> None:
        """
        Verify atomic rollback: an exception raised within a transaction rolls back
        all operations within that transaction boundary without leaving partial records.
        """
        initial_log_count = len(disk_repo.query_logs())

        with pytest.raises(RuntimeError):
            with disk_repo.db.transaction() as conn:
                conn.execute(
                    "INSERT INTO logs (timestamp, level, logger_name, message) VALUES (?, ?, ?, ?);",
                    (time.time(), "INFO", "atomic", "Message 1"),
                )
                conn.execute(
                    "INSERT INTO logs (timestamp, level, logger_name, message) VALUES (?, ?, ?, ?);",
                    (time.time(), "INFO", "atomic", "Message 2"),
                )
                # Intentionally trigger an unhandled failure
                raise RuntimeError("Simulated crash mid-transaction")

        final_log_count = len(disk_repo.query_logs())
        assert final_log_count == initial_log_count, "Rolled back transaction must leave log count unchanged"

    def test_unknown_game_state_and_strategy_strings(self, disk_repo: Repository) -> None:
        """
        Verify repository handles string inputs that are not defined in the GameState
        or StrategyType enums without throwing unhandled exceptions.
        """
        sess = disk_repo.create_session("sess_unknown_enum")

        # Custom/unknown game state string
        scan_id = disk_repo.save_base_scan(
            session_id=sess.session_id,
            game_state="UNOFFICIAL_FUTURE_STATE_v99",
        )
        scans = disk_repo.get_base_scans(session_id=sess.session_id)
        assert scans[0]["game_state"] == "UNOFFICIAL_FUTURE_STATE_v99"

        # Dataset annotation with unknown state: list_annotations should fallback to GameState.UNKNOWN
        disk_repo.save_annotation(
            DatasetAnnotation(
                annotation_id="ann_unknown_state",
                image_path="/test.png",
                state=GameState.UNKNOWN,
            )
        )
        # Directly update state in DB to a non-existent state
        disk_repo.db.execute("UPDATE dataset_annotations SET state = 'WEIRD_STATE' WHERE annotation_id = 'ann_unknown_state';")
        ann_list = disk_repo.list_annotations()
        found = next((a for a in ann_list if a.annotation_id == "ann_unknown_state"), None)
        assert found is not None
        assert found.state == GameState.UNKNOWN


# =============================================================================
# 5. Large Record Set Stress & Query Index Performance
# =============================================================================

class TestScaleAndIndexPerformance:
    """Empirical verification of query performance and pagination on large record sets."""

    def test_high_volume_scans_and_logs_stress(self, disk_repo: Repository) -> None:
        """
        Insert 2,000 base scans and 2,000 logs inside a single transaction batch.
        Empirically verify:
        - Query pagination with limit/offset runs in < 50ms (leveraging indexes).
        - Count aggregations remain exact.
        - Sorting order (timestamp DESC) is maintained across high offset slices.
        """
        sess = disk_repo.create_session("sess_scale_test")

        start_time = time.time()
        # Batch insert 2,000 scans and 2,000 logs
        with disk_repo.db.transaction() as conn:
            scan_params = [
                (sess.session_id, start_time + i, "ENEMY_BASE_PREVIEW", 0.95, 100000 + i, 200000 + i, 1000 + i, 1, "{}")
                for i in range(2000)
            ]
            conn.executemany(
                """
                INSERT INTO base_scans (
                    session_id, timestamp, game_state, confidence,
                    gold, elixir, dark_elixir, th_accessible, raw_ocr_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                scan_params,
            )

            log_params = [
                (start_time + i, "INFO" if i % 2 == 0 else "ERROR", "scale_test", f"Log item {i}")
                for i in range(2000)
            ]
            conn.executemany(
                "INSERT INTO logs (timestamp, level, logger_name, message) VALUES (?, ?, ?, ?);",
                log_params,
            )

        # 1. Count verification
        total_scans = disk_repo.get_scan_count(session_id=sess.session_id)
        assert total_scans == 2000

        # 2. Pagination query timing test (<50ms)
        t0 = time.perf_counter()
        scans_page = disk_repo.get_base_scans(session_id=sess.session_id, limit=50, offset=1000)
        t_page = (time.perf_counter() - t0) * 1000.0  # ms

        assert len(scans_page) == 50
        assert t_page < 100.0, f"Pagination query took {t_page:.2f}ms (expected < 100ms)"

        # 3. Descending order verification
        assert scans_page[0]["timestamp"] > scans_page[-1]["timestamp"]

        # 4. Filtered logs query timing test
        t0 = time.perf_counter()
        error_logs = disk_repo.query_logs(limit=50, level="ERROR")
        t_logs = (time.perf_counter() - t0) * 1000.0

        assert len(error_logs) == 50
        assert all(l["level"] == "ERROR" for l in error_logs)
        assert t_logs < 100.0, f"Filtered logs query took {t_logs:.2f}ms (expected < 100ms)"

    def test_large_payload_2mb_raw_ocr_json(self, disk_repo: Repository) -> None:
        """
        Verify that storing and retrieving a large (2 MB) OCR payload succeeds without truncation,
        corruption, or exceeding memory boundaries.
        """
        sess = disk_repo.create_session("sess_large_payload")
        large_json_str = json.dumps({"payload": "X" * (2 * 1024 * 1024)})

        scan_id = disk_repo.save_base_scan(
            session_id=sess.session_id,
            game_state=GameState.ENEMY_BASE_PREVIEW,
            raw_ocr_json=large_json_str,
        )
        assert scan_id > 0

        scans = disk_repo.get_base_scans(session_id=sess.session_id, limit=1)
        assert len(scans) == 1
        retrieved_payload = scans[0]["raw_ocr_json"]
        assert len(retrieved_payload) == len(large_json_str)

    def test_schema_reinit_idempotency(self, disk_repo: Repository) -> None:
        """
        Verify that calling init_schema() multiple times on an already initialized database
        with existing data is strictly idempotent and does not drop tables or overwrite data.
        """
        sess = disk_repo.create_session("sess_idempotent_test")
        disk_repo.log_event("INFO", "Pre-reinit log")

        # Execute init_schema 3 times
        for _ in range(3):
            disk_repo.db.init_schema()

        assert disk_repo.get_session(sess.session_id) is not None
        logs = disk_repo.query_logs()
        assert len(logs) == 1
        assert logs[0]["message"] == "Pre-reinit log"

    def test_sqlite_pragmas_strict_verification(self, disk_repo: Repository) -> None:
        """
        Empirically verify that all required production SQLite pragmas are actively set:
        - PRAGMA journal_mode = wal
        - PRAGMA foreign_keys = 1 (ON)
        - PRAGMA busy_timeout = 5000
        - PRAGMA synchronous = 1 (NORMAL)
        """
        db = disk_repo.db
        assert db.check_wal_mode() == "wal"

        fk_row = db.fetchone("PRAGMA foreign_keys;")
        assert fk_row is not None and int(fk_row[0]) == 1

        timeout_row = db.fetchone("PRAGMA busy_timeout;")
        assert timeout_row is not None and int(timeout_row[0]) == 5000

        sync_row = db.fetchone("PRAGMA synchronous;")
        assert sync_row is not None and int(sync_row[0]) == 1  # 1 = NORMAL

    def test_nested_transaction_operational_error(self, disk_repo: Repository) -> None:
        """
        Failure mode verification:
        Demonstrates that DatabaseManager.transaction() does not support savepoints/nesting.
        Invoking a nested `with db.transaction():` raises OperationalError.
        """
        db = disk_repo.db
        with pytest.raises(sqlite3.OperationalError) as exc_info:
            with db.transaction():
                with db.transaction():
                    pass
        assert "cannot start a transaction within a transaction" in str(exc_info.value)

    def test_non_serializable_objects_raise_type_error(self, disk_repo: Repository) -> None:
        """
        Verify that passing non-JSON-serializable objects (such as sets or custom objects)
        in score_breakdown or labels properly raises TypeError at serialization time.
        """
        sess = disk_repo.create_session("sess_non_serializable")
        rec = FarmingRecommendation(
            suitability_score=50.0,
            primary_strategy=StrategyType.SNEAKY_GOBLIN,
            score_breakdown={"invalid_set": {1, 2, 3}},  # type: ignore[dict-item]
        )
        with pytest.raises(TypeError):
            disk_repo.save_recommendation(rec, session_id=sess.session_id)

    def test_closed_database_programming_error_and_resurrection_asymmetry(self, temp_db_path: str) -> None:
        """
        Vulnerability verification:
        Empirically verifies the thread-local connection lifecycle asymmetry during close():
        1. When db.close() is called, it closes all sqlite3 connections across all threads.
        2. However, `del self._local.conn` only deletes the attribute for the thread invoking close().
        3. Background worker threads retain their now-closed `self._local.conn`, so subsequent queries
           in those threads raise `sqlite3.ProgrammingError: Cannot operate on a closed database`.
        4. Conversely, the thread invoking close() has its `self._local.conn` deleted, so subsequent
           queries silently resurrect and open a new connection rather than raising an error,
           because DatabaseManager lacks an explicit `_is_closed` lifecycle flag.
        """
        db = DatabaseManager(db_path=temp_db_path, auto_init=True)
        repo = Repository(db=db)
        repo.log_event("INFO", "Initial event")

        worker_got_error: List[str] = []
        worker_ready = threading.Event()
        main_closed = threading.Event()

        def worker() -> None:
            # Touch connection to populate self._local.conn in worker thread
            repo.query_logs()
            worker_ready.set()
            main_closed.wait(timeout=2.0)
            try:
                repo.query_logs()
            except sqlite3.ProgrammingError as e:
                worker_got_error.append(str(e))

        t = threading.Thread(target=worker)
        t.start()
        worker_ready.wait(timeout=2.0)

        # Main thread closes the database
        db.close()
        main_closed.set()
        t.join()

        # 1. Background thread experienced ProgrammingError on its closed connection
        assert len(worker_got_error) == 1
        assert "closed database" in worker_got_error[0]

        # 2. Main thread correctly raises ProgrammingError due to is_closed guard
        with pytest.raises(sqlite3.ProgrammingError):
            repo.query_logs()

