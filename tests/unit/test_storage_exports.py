"""
Unit tests for ClashVision Data Repository Export Utilities.
Verifies Phase 8 Requirements:
- export_sessions_to_json
- export_scans_to_csv
- export_battles_to_json
- export_analytics_to_csv
"""

import json
import os
import tempfile
import unittest

from clashvision.core.constants import GameState
from clashvision.storage.db import DatabaseManager
from clashvision.storage.repository import Repository


class TestStorageExports(unittest.TestCase):
    """Verifies all JSON and CSV export endpoints on Repository."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_export.db")
        self.db = DatabaseManager(db_path=self.db_path)
        self.repo = Repository(db=self.db)

        # Seed test data
        self.sess1 = self.repo.create_session("sess_alpha", notes="Alpha Test Session")
        self.repo.save_base_scan(
            session_id="sess_alpha",
            timestamp=1000.0,
            game_state=GameState.ENEMY_BASE_PREVIEW,
            confidence=0.95,
            gold=850_000,
            elixir=750_000,
            dark_elixir=6_000,
            th_accessible=True,
            raw_ocr_json={"battle_result": {"stars": 2, "destruction_percent": 75}},
        )
        self.repo.save_base_scan(
            session_id="sess_alpha",
            timestamp=1050.0,
            game_state=GameState.ENEMY_BASE_PREVIEW,
            confidence=0.92,
            gold=950_000,
            elixir=850_000,
            dark_elixir=7_000,
            th_accessible=False,
            raw_ocr_json={"battle_result": {"stars": 3, "destruction_percent": 100}},
        )
        self.repo.end_session("sess_alpha", end_time=1200.0)

    def tearDown(self):
        self.db.close()
        self.temp_dir.cleanup()

    def test_export_sessions_to_json(self):
        """Verifies session export returns valid JSON and writes to disk."""
        out_file = os.path.join(self.temp_dir.name, "sessions.json")
        json_str = self.repo.export_sessions_to_json(output_path=out_file)

        self.assertTrue(os.path.exists(out_file))
        data = json.loads(json_str)
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)
        self.assertEqual(data[0]["session_id"], "sess_alpha")
        self.assertEqual(data[0]["notes"], "Alpha Test Session")

    def test_export_scans_to_csv(self):
        """Verifies base scans are serialized into standard CSV format."""
        out_file = os.path.join(self.temp_dir.name, "scans.csv")
        count = self.repo.export_scans_to_csv(output_path=out_file, session_id="sess_alpha")

        self.assertEqual(count, 2)
        self.assertTrue(os.path.exists(out_file))
        with open(out_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)  # 1 header + 2 rows
        self.assertIn("scan_id,session_id,timestamp", lines[0])
        self.assertIn("850000", lines[1])

    def test_export_battles_to_json(self):
        """Verifies battle simulation records extract correctly."""
        out_file = os.path.join(self.temp_dir.name, "battles.json")
        json_str = self.repo.export_battles_to_json(output_path=out_file, session_id="sess_alpha")

        self.assertTrue(os.path.exists(out_file))
        battles = json.loads(json_str)
        self.assertEqual(len(battles), 2)
        self.assertEqual(battles[0]["battle_result"]["stars"], 2)
        self.assertEqual(battles[1]["battle_result"]["destruction_percent"], 100)

    def test_export_analytics_to_csv(self):
        """Verifies session aggregate analytics CSV generation."""
        out_file = os.path.join(self.temp_dir.name, "analytics.csv")
        rows = self.repo.export_analytics_to_csv(output_path=out_file)

        self.assertGreaterEqual(rows, 1)
        self.assertTrue(os.path.exists(out_file))
        with open(out_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("sess_alpha", content)
        self.assertIn("1800000", content)  # 850k + 950k Gold


if __name__ == "__main__":
    unittest.main()

