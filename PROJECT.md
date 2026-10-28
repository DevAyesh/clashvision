# Project: ClashVision — Intelligent Farming & Tactical Analysis Dashboard

## Architecture

ClashVision is architected around a layered, decoupled design separating OS-level capture, computer vision, decision intelligence, sandboxed simulation, local storage, and the presentation layer. Cross-component communication uses typed dataclass contracts and asynchronous Qt signal-slot channels, guaranteeing a 60 FPS non-blocking UI.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PySide6 / PyQt6 UI Layer                        │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │ 9 Dark-Themed Views: Overview, Live Capture, Base Analyzer,        │ │
│ │ Strategy Planner, Battle Simulator, Analytics, Dataset Manager,    │ │
│ │ Settings, Logs                                                     │ │
│ └────────────────────────────────────────────────────────────────────┘ │
└───────────────────▲───────────────────────────────▲────────────────────┘
                    │ Qt Signals (QueuedConnection) │
┌───────────────────┴───────────────────────────────┴────────────────────┐
│                    Asynchronous Worker Thread Pool                     │
│ ┌───────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────────┐ │
│ │ CaptureWorker │ │ VisionWorker │ │  OCRWorker   │ │ DecisionWorker │ │
│ └───────▲───────┘ └──────▲───────┘ └──────▲───────┘ └────────▲───────┘ │
└─────────┼────────────────┼────────────────┼──────────────────┼─────────┘
          │                │                │                  │
┌─────────┴────────┐ ┌─────┴──────┐ ┌───────┴──────┐ ┌─────────┴───────┐
│ Capture Subsystem│ │Vision State│ │  Pluggable   │ │Farming Decision │
│ - Win32 Detector │ │ Classifier │ │  OCR Engine  │ │& Strategy Engine│
│ - MSS Grabber    │ │ (12 States)│ │ (Tess/Rapid/ │ │(5 Strategies:   │
│ - Coord Transform│ │ - Ensemble │ │     Mock)    │ │ Sneaky, BARCH,  │
│ - Read-Only Safe │ │ - Debouncer│ │ - Sanitizer  │ │ Miner, Dragon,  │
│                  │ │ - Evidence │ │ - Formatting │ │ Custom)         │
└──────────────────┘ └────────────┘ └──────────────┘ └─────────────────┘
                             ▲                              ▲
┌────────────────────────────┴──────────────────────────────┴──────────┐
│                   Local Attack Simulator Subsystem                    │
│ - Strict Isolation Barrier: PyAutoGUI isolated ONLY to simulator     │
│ - Playback Controls: Start, Pause, Stop, Step, Reset, Replay         │
│ - Emergency Stop: Hardware/software interrupt <= 50ms latency        │
│ - Safety Enforcer: Rejects any live window interaction               │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────┐
│                 SQLite Storage & Analytics Subsystem                 │
│ - WAL Mode Relational Database (7 Tables)                            │
│ - Sessions, Base Scans, Decisions, Evaluations, Settings, Logs       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Feature Inventory

Every feature surveyed across R1–R6 is cataloged below with its designated milestone.

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Window Enumeration & Target Discovery | Heuristic process and window detection for Google Play Games PC (`PlayGamesPC.exe`, `HD-Player.exe`, `crosvm.exe`, title regex). | M1 | R1 |
| 2 | High-Performance MSS Frame Capture | Non-blocking frame capture using `mss` with target framerate (10–15 FPS) and low CPU utilization. | M1 | R1 |
| 3 | DPI Scaling & Dynamic Resize Transform | Win32 Per-Monitor DPI awareness and coordinate normalization between client pixels and normalized virtual coords. | M1 | R1 |
| 4 | Window State & Visibility Monitor | Detect minimized, occluded, or background state, emitting status warnings and auto-pausing capture. | M1 | R1 |
| 5 | Read-Only Live Safety Guard | Hard architectural barrier preventing any click/keystroke injection toward the live emulator window. | M1 | R1, R4 |
| 6 | 12 Game State Classifier | Classify all 12 states (Home village, Multiplayer, Matchmaking, Enemy base preview, Battle in progress, Battle result, Training, Army config, Loading, Connection error, Popup, Unknown). | M2 | R2 |
| 7 | State Classifier Evidence & Confidence | Multi-feature heuristic ensemble (color histograms, template anchors, UI landmarks) outputting confidence $\in [0.0, 1.0]$ and explanatory evidence. | M2 | R2 |
| 8 | Multi-Frame State Debouncing | 2-frame temporal consistency filter eliminating transient classification noise and flicker. | M2 | R2 |
| 9 | Resource ROI Localization | Dynamic extraction of Gold, Elixir, Dark Elixir, troop count, and battle timer bounding boxes across resolutions. | M2 | R2 |
| 10 | Vision Image Preprocessing | Grayscale conversion, adaptive thresholding, morphological noise removal, and contrast enhancement for text OCR. | M2 | R2 |
| 11 | Pluggable OCR Engine Abstraction | Unified OCR interface supporting Pytesseract, RapidOCR fallback, and deterministic Mock engine. | M2 | R2 |
| 12 | Numeric OCR Sanitizer & Value Parser | Parse standard ("800,000", "800K", "1.2M", "800.000") and timer formats into clean integers; zero hallucination on unread text. | M2 | R2 |
| 13 | Visual OCR Debug Data Model | Data structure capturing raw ROI crop, preprocessed thresholded image, raw OCR string, and parsed integer. | M2 | R2 |
| 14 | Configurable Farming Base Evaluator | Base evaluation scoring algorithm weighting Gold, Elixir, and Dark Elixir against user-defined thresholds. | M3 | R3 |
| 15 | Base Layout Accessibility Analysis | Evaluate exposed collectors vs core storages, perimeter defenses, and Town Hall accessibility. | M3 | R3 |
| 16 | Strategy A: Sneaky Goblin Farming | Collector/storage routing analysis and Town Hall sniping viability heuristic. | M3 | R3 |
| 17 | Strategy B: BARCH Farming | Outer ring building clearing and perimeter collector sweep evaluation. | M3 | R3 |
| 18 | Strategy C: Mass Miner | Compartment path analysis, core routing, and spell placement recommendations. | M3 | R3 |
| 19 | Strategy D: Mass Dragon | Air defense positioning analysis, sweeper orientation, and air attack corridor calculation. | M3 | R3 |
| 20 | Strategy E: Custom User Strategy | Modular strategy engine accepting user army composition, custom thresholds, and priority weightings. | M3 | R3 |
| 21 | Explainable Recommendation Breakdown | Detailed risk vs reward assessment, score breakdown, prerequisite checks, and deployment sequencing. | M3 | R3 |
| 22 | Local Attack Simulator Engine | Offline battle simulation engine operating on recorded/synthetic screenshot fixtures. | M4 | R4 |
| 23 | Troop Deployment & Path Simulation | Simulated troop deployment markers, delay timers, and timeline playback. | M4 | R4 |
| 24 | Simulator Playback Controls | Start, pause, stop, step-forward, and reset controls for simulated replay. | M4 | R4 |
| 25 | Simulator Sandboxed Input Isolation | Strict sandboxing of PyAutoGUI controls exclusively within the simulator canvas/window bounds. | M4 | R4 |
| 26 | Emergency Stop Safety Control | Prominent hardware hotkey and UI emergency stop button halting playback within $\le 50\text{ms}$. | M4 | R4 |
| 27 | Visual Mode Safety Indicator | Distinct, prominent status banner clearly differentiating Live Observation (Read-Only) from Local Simulation. | M4 | R4 |
| 28 | Dark-Themed PySide6 Dashboard Shell | Professional responsive dark stylesheet, sidebar navigation, and header status bar. | M5 | R5 |
| 29 | View 1: Overview Dashboard | Live system status, current detected state, resources, recommendation summary, error panel. | M5 | R5 |
| 30 | View 2: Live Capture Feed | Live feed with toggleable bounding boxes, detected UI elements, and confidence badges. | M5 | R5 |
| 31 | View 3: Base Analyzer | Annotated base overlay breakdown, resource distribution, and suitability score. | M5 | R5 |
| 32 | View 4: Strategy Planner | Side-by-side strategy comparison, required troops, and step-by-step deployment plan. | M5 | R5 |
| 33 | View 5: Battle Simulator | Interactive battle playback viewer with troop markers and timeline scrubber. | M5 | R5 |
| 34 | View 6: Analytics Dashboard | Historical charts and metrics (bases scanned, target match rate, OCR accuracy, CV latency). | M5 | R5 |
| 35 | View 7: Dataset Manager | Screenshot library with labeling tools, bounding box annotation, and JSON export. | M5 | R5 |
| 36 | View 8: Settings View | Configuration for capture rate, OCR paths, resource thresholds, and log levels. | M5 | R5 |
| 37 | View 9: Structured Logs View | Live virtualized log viewer with log-level filtering and search. | M5 | R5 |
| 38 | SQLite WAL Persistence Layer | 7-table schema persisting sessions, scans, decisions, evaluations, settings, logs, and datasets. | M5 | R5 |
| 39 | Non-Blocking QThread Worker Pool | Background QThread workers ensuring the Qt GUI event loop never blocks during capture/CV/OCR. | M5 | R5 |
| 40 | Offline Synthetic Fixture Suite | Generators producing pixel-accurate 12-state frames, OCR resource crops, and 6 base layouts. | M6 | R6 |
| 41 | 4-Tier Automated PyTest Suite | 100% offline test suite covering Tiers 1–4 with >=5 tests per feature. | M6 | R6 |
| 42 | Tier 5 Adversarial Coverage Hardening | White-box stress tests, extreme boundaries, and mutation resistance verified by Challengers. | M6 | R6 |
| 43 | Local Git Repository & Milestone Commits | Initialized Git repository with structured milestone commits without pushing to remote. | M7 | R6 |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Core Capture & Safe Window Detection | Windows detection, MSS capture, DPI coordinate transforms, read-only safety guard. | none | DONE |
| M2 | Vision State Recognition & OCR Engine | 12 game states, ensemble evidence, debouncing, OCR parser ("800K", "1.2M"), debug models. | M1 | DONE |
| M3 | Farming Decision & Multi-Strategy Engine | Base evaluator, layout analysis, 5 strategies (Sneaky, BARCH, Miner, Dragon, Custom), explainability. | M2 | DONE |
| M4 | Local Attack Simulator & Safety Sandboxing | Fixture replay, deployment simulation, PyAutoGUI sandbox isolation, emergency stop hotkey. | M3 | PLANNED |
| M5 | PySide6 Dark Dashboard & SQLite Storage | 9-view dark UI, async QThread workers, WAL SQLite persistence, settings manager. | M4 | PLANNED |
| M6 | E2E Integration, 100% Test Pass & Tier 5 | Run complete offline pytest suite (Tiers 1–4), pass 100%, and apply Tier 5 adversarial hardening. | M5, Test Track | PLANNED |
| M7 | Local Git Init & Milestone Commits | Initialize local Git repository and create clean milestone commits (no remote push). | M6 | PLANNED |

---

## Interface Contracts

### 1. `CaptureSubsystem` ↔ `VisionSubsystem`
- **Data Contract**: `clashvision.core.models.FramePacket`
  - `frame: np.ndarray` (BGR image, shape `(H, W, 3)`)
  - `timestamp: float` (POSIX timestamp)
  - `window_info: WindowInfo` (`is_visible: bool`, `client_rect: Rect`, `scale_factor: float`)
  - `frame_id: int` (Monotonically increasing)

### 2. `VisionSubsystem` ↔ `DecisionSubsystem`
- **Data Contract**: `clashvision.core.models.VisionAnalysisResult`
  - `state: GameState` (Enum: 12 states)
  - `confidence: float` ($[0.0, 1.0]$)
  - `evidence: Dict[str, Any]` (Detected UI landmarks, anchors, histograms)
  - `resources: Optional[ResourceReading]` (`gold: Optional[int]`, `elixir: Optional[int]`, `dark_elixir: Optional[int]`, `raw_text: Dict[str, str]`)
  - `base_layout: Optional[BaseLayoutFeatures]` (`exposed_collector_ratio: float`, `th_accessible: bool`, `air_defense_locations: List[Tuple[int, int]]`)

### 3. `DecisionSubsystem` ↔ `SimulatorSubsystem` & `PresentationSubsystem`
- **Data Contract**: `clashvision.core.models.FarmingRecommendation`
  - `suitability_score: float` ($[0.0, 100.0]$)
  - `primary_strategy: StrategyType` (`SNEAKY_GOBLIN`, `BARCH`, `MASS_MINER`, `MASS_DRAGON`, `CUSTOM`)
  - `rankings: List[StrategyEvaluation]` (Per-strategy score, viability, and risk)
  - `deployment_plan: List[DeploymentStep]` (Troop type, target coordinate, delay ms)
  - `score_breakdown: Dict[str, float]` (Resource score, accessibility score, risk discount)

### 4. `SimulatorSubsystem` ↔ Safety Enforcer
- **Contract**: `SandboxedInputController.verify_and_execute(action: InputAction)`
  - Raises `LiveInputSafetyViolation` if target window is NOT the simulator canvas.
  - Aborts immediately if `emergency_stop_flag.is_set()`.

### 5. Backend Workers ↔ UI Presentation Layer
- **Contract**: Qt Queued Signals
  - `sig_frame_captured(FramePacket)`
  - `sig_state_detected(GameState, float, dict)`
  - `sig_ocr_parsed(ResourceReading)`
  - `sig_decision_ready(FarmingRecommendation)`
  - `sig_sim_tick(SimulationState)`
  - `sig_error(str, str)`

---

## Code Layout

```
clashvision/
├── __init__.py
├── __main__.py
├── core/
│   ├── __init__.py
│   ├── config.py                 # Application configuration & user settings dataclasses
│   ├── constants.py              # 12 GameState enums, StrategyType enums, UI constants
│   ├── coordinates.py            # Resolution & DPI coordinate transforms
│   ├── exceptions.py             # Custom exceptions & LiveInputSafetyViolation
│   └── models.py                 # Pure dataclasses (FramePacket, ResourceReading, etc.)
├── capture/
│   ├── __init__.py
│   ├── window_detector.py        # Windows process/window enumeration & DPI awareness
│   ├── mss_grabber.py            # High-performance MSS screen capture engine
│   └── safety_guard.py           # Read-only enforcement barrier (zero input to emulator)
├── vision/
│   ├── __init__.py
│   ├── state_classifier.py       # 12 Game state classification ensemble & debouncing
│   ├── ocr_engine.py             # Pluggable OCR interface (Pytesseract, RapidOCR, Mock)
│   ├── numeric_parser.py         # Parsing for "800,000", "800K", "1.2M", timer formats
│   ├── preprocessor.py           # Adaptive thresholding, ROI cropping, morphology
│   └── layout_analyzer.py        # Base layout feature extraction (collectors, TH, AD)
├── decision/
│   ├── __init__.py
│   ├── evaluator.py              # Base resource scoring & threshold comparison
│   ├── base_strategy.py          # Abstract strategy interface & deployment models
│   ├── sneaky_goblin.py          # Strategy A: Sneaky Goblin Farming
│   ├── barch.py                  # Strategy B: BARCH Farming
│   ├── mass_miner.py             # Strategy C: Mass Miner
│   ├── mass_dragon.py            # Strategy D: Mass Dragon
│   ├── custom_strategy.py        # Strategy E: User-defined Custom Strategy
│   └── recommendation_engine.py  # Ranking, risk assessment & explainability
├── simulator/
│   ├── __init__.py
│   ├── simulation_engine.py      # Offline playback & troop deployment state machine
│   ├── sandboxed_controller.py   # Quarantined PyAutoGUI controller (simulator canvas only)
│   ├── safety_barrier.py         # Emergency stop hotkey & window handle verification
│   └── replay_recorder.py        # Playback timeline recording and stepping
├── storage/
│   ├── __init__.py
│   ├── db.py                     # SQLite WAL connection manager & migrations
│   ├── repository.py             # CRUD operations for sessions, scans, settings, logs
│   └── schema.sql                # 7-table relational schema definition
└── ui/
    ├── __init__.py
    ├── qt_compat.py              # Transparent PySide6 / PyQt6 compatibility shim
    ├── main_window.py            # Main application window & dark sidebar layout
    ├── styles.py                 # Dark modern QSS theme
    ├── workers.py                # Asynchronous QThread workers
    └── views/
        ├── __init__.py
        ├── overview_view.py      # View 1: Overview
        ├── live_capture_view.py  # View 2: Live Capture Feed & Overlays
        ├── base_analyzer_view.py # View 3: Base Analyzer
        ├── strategy_view.py      # View 4: Strategy Planner
        ├── simulator_view.py     # View 5: Battle Simulator
        ├── analytics_view.py     # View 6: Analytics Dashboard
        ├── dataset_view.py       # View 7: Screenshot Dataset Manager
        ├── settings_view.py      # View 8: Settings View
        └── logs_view.py          # View 9: Structured Logs View

tests/
├── __init__.py
├── conftest.py                   # Pytest fixtures, mock OCR, safety interceptor
├── run_tests.py                  # Zero-dependency test runner script
├── fixtures/
│   ├── __init__.py
│   ├── synthetic_generator.py    # Programmatic generator for 12 states & OCR crops
│   ├── layout_generator.py       # Programmatic generator for 6 base layout graphs
│   └── recorded_samples.py       # Ground truth recorded test fixtures
├── unit/
│   ├── test_coordinates.py
│   ├── test_capture_safety.py
│   ├── test_state_classifier.py
│   ├── test_numeric_parser.py
│   ├── test_ocr_engine.py
│   ├── test_decision_engine.py
│   ├── test_strategies.py
│   ├── test_simulator_sandbox.py
│   └── test_storage.py
├── integration/
│   ├── test_pipeline_e2e.py      # Full pipeline: Capture -> Vision -> OCR -> Decision
│   ├── test_simulator_e2e.py     # Simulator playback & emergency stop e2e
│   └── test_ui_headless.py       # Headless UI model/worker signal verification
└── tiers/
    ├── test_tier1_features.py    # Tier 1: Feature Coverage (>=5 per feature)
    ├── test_tier2_boundaries.py  # Tier 2: Boundary & Corner Cases (>=5 per feature)
    ├── test_tier3_pairwise.py    # Tier 3: Pairwise Combinations
    └── test_tier4_workloads.py   # Tier 4: Real-World Application Workloads
```
