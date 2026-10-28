-- ClashVision Relational SQLite Storage Schema
-- Supports SQLite WAL persistence mode with PRAGMA foreign_keys = ON

PRAGMA foreign_keys = ON;

-- 1. Sessions table: Tracks distinct observation/simulation application sessions
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    start_time REAL NOT NULL,
    end_time REAL,
    total_scans INT DEFAULT 0,
    matched_targets INT DEFAULT 0,
    notes TEXT
);

-- 2. Base Scans table: Captured frames and vision classification records
CREATE TABLE IF NOT EXISTS base_scans (
    scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    timestamp REAL NOT NULL,
    game_state TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    gold INT,
    elixir INT,
    dark_elixir INT,
    th_accessible INT DEFAULT 0,
    raw_ocr_json TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE SET NULL
);

-- 3. Decisions table: Farming recommendation decisions emitted for bases
CREATE TABLE IF NOT EXISTS decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INT,
    session_id TEXT,
    timestamp REAL NOT NULL,
    primary_strategy TEXT NOT NULL,
    suitability_score REAL DEFAULT 0.0,
    explanation TEXT,
    breakdown_json TEXT,
    FOREIGN KEY (scan_id) REFERENCES base_scans(scan_id) ON DELETE SET NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE SET NULL
);

-- 4. Evaluations table: Per-strategy candidate evaluation scores and viability
CREATE TABLE IF NOT EXISTS evaluations (
    eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INT NOT NULL,
    strategy_type TEXT NOT NULL,
    score REAL DEFAULT 0.0,
    viable INT DEFAULT 0,
    risk_level TEXT,
    reasons_json TEXT,
    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id) ON DELETE CASCADE
);

-- 5. Settings table: Key-value persistence for application configurations
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

-- 6. Logs table: Structured application logging records
CREATE TABLE IF NOT EXISTS logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    level TEXT NOT NULL,
    logger_name TEXT NOT NULL,
    message TEXT NOT NULL
);

-- 7. Dataset Annotations table: Curated ground-truth frame annotations
CREATE TABLE IF NOT EXISTS dataset_annotations (
    annotation_id TEXT PRIMARY KEY,
    image_path TEXT NOT NULL,
    timestamp REAL NOT NULL,
    state TEXT NOT NULL,
    labels_json TEXT,
    bounding_boxes_json TEXT
);

-- High-performance query indexes
CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_base_scans_session_id ON base_scans(session_id);
CREATE INDEX IF NOT EXISTS idx_base_scans_timestamp ON base_scans(timestamp);
CREATE INDEX IF NOT EXISTS idx_base_scans_state ON base_scans(game_state);
CREATE INDEX IF NOT EXISTS idx_decisions_scan_id ON decisions(scan_id);
CREATE INDEX IF NOT EXISTS idx_decisions_session_id ON decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);
CREATE INDEX IF NOT EXISTS idx_evaluations_decision_id ON evaluations(decision_id);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level);
CREATE INDEX IF NOT EXISTS idx_dataset_annotations_state ON dataset_annotations(state);
CREATE INDEX IF NOT EXISTS idx_dataset_annotations_timestamp ON dataset_annotations(timestamp);
