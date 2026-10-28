# ClashVision: Computer Vision Farming Assistant for Clash of Clans

**ClashVision** is an offline-capable, computer-vision-driven analytics, farming recommendation, and battle simulation desktop application designed for Clash of Clans running on **Google Play Games PC** on Windows 11.

---

## 1. Application Purpose & Operating Principles

ClashVision analyzes battles, evaluates resource farming opportunities, recommends attack strategies based on base topology, and provides a sandboxed local deployment simulator.

### Core Safety Invariants
- **Read-Only Live Observation**: Captures frames from the Google Play Games PC window for visual analysis only. Zero automated mouse clicks or keystrokes are sent to the game client.
- **Isolated Local Simulator**: Automated deployment playback is restricted exclusively to the local simulator canvas operating over synthetic or recorded screenshot fixtures.
- **No Game Client Tampering**: Does not read game memory, hook binaries, intercept network packets, modify game files, or bypass game protection.
- **No Account Automation**: Never automates gem spending, purchases, or irreversible account actions.
- **Sub-millisecond Emergency Stop**: An active emergency stop circuit halts local simulation immediately (< 0.2ms latency).

---

## 2. System Architecture

The application is structured into decoupled, modular subsystems:

```
clashvision/
├── capture/                  # Screen capture & window detection
│   ├── window_detector.py    # Win32 DPI-aware window detection & bounding
│   ├── mss_grabber.py        # High-performance MSS capture loop with rate pacing
│   └── safety_guard.py       # Strict input quarantine barrier & emergency stop
├── vision/                   # Computer vision & OCR
│   ├── state_classifier.py   # 12-state ensemble classifier with temporal debouncing
│   ├── numeric_parser.py     # Deterministic number, compact (K/M), and timer parser
│   ├── ocr_engine.py         # Pluggable Tesseract, RapidOCR, and MockOCR backends
│   ├── preprocessor.py       # CLAHE contrast enhancement, adaptive thresholding, ROI cropping
│   └── layout_analyzer.py    # Base collector exposure, storage positioning & defense vectoring
├── decision/                 # Farming scoring & attack recommendations
│   ├── evaluator.py          # Multi-factor loot efficiency & risk scoring [0.0, 100.0]
│   ├── recommendation_engine.py # Strategy ranking with explainable rationale
│   ├── sneaky_goblin.py      # Strategy A: Resource collector & Town Hall sniping
│   ├── barch.py              # Strategy B: Outer-ring building & collector sweeps
│   ├── mass_miner.py         # Strategy C: Compartment navigation avoiding splash clusters
│   ├── mass_dragon.py        # Strategy D: Air defense corridor analysis & funneling
│   └── custom_strategy.py    # Strategy E: User-defined army compositions and rules
├── simulator/                # Local battle replay & sandboxing
│   ├── simulation_engine.py  # 5-state playback engine (IDLE, PLAYING, PAUSED, COMPLETED, RESET)
│   ├── autonomous_state_machine.py # 15-state event-driven state machine with transition graph
│   ├── target_selector.py    # Multi-mode loot filter (ALL, ANY, WEIGHTED, CUSTOM) & scoring
│   ├── deployment_executor.py # Paced simulated deployment executor with safety verification
│   ├── battle_resolver.py    # Multi-strategy outcome calculation, star attribution, & loot resolution
│   ├── autonomous_engine.py  # End-to-end full-cycle autonomous farming engine
│   ├── sandboxed_controller.py # Quarantined coordinate controller with LiveInputSafetyViolation
│   ├── safety_barrier.py     # Sub-millisecond emergency stop latching
│   └── replay_recorder.py    # JSON timeline recording and deterministic replaying
├── storage/                  # SQLite WAL database & repository
├── db.py                 # SQLite WAL connection manager with thread-safe locking
│   └── repository.py         # CRUD repository for sessions, scans, decisions, logs, and datasets
├── ui/                       # PySide6 dark-themed desktop dashboard
│   ├── main_window.py        # Primary shell with sidebar navigation & emergency stop
│   ├── styles.py             # Dark theme palette & CSS stylesheet
│   ├── workers.py            # Non-blocking asynchronous QThread background workers
│   ├── qt_compat.py          # PySide6/PyQt6 abstraction with headless test fallback
│   └── views/                # 10 functional views
│       ├── overview_view.py       # Live system overview, resources, and alerts
│       ├── live_capture_view.py   # Real-time preview with toggleable CV overlays
│       ├── base_analyzer_view.py  # Annotated base breakdown & suitability meters
│       ├── strategy_view.py       # Tactical plan visualizer & deployment steps
│       ├── simulator_view.py      # Manual simulation sandbox & interactive controls
│       ├── autonomous_simulator_view.py # Full-cycle autonomous farming dashboard
│       ├── sessions_view.py       # Historical farming sessions & analytics
│       ├── replay_view.py         # Frame-by-frame attack replay inspector
│       ├── settings_view.py       # Loot thresholds, OCR confidence, & engine tuning
│       └── system_health_view.py  # FPS meters, memory monitor, & latency gauges
│       ├── strategy_view.py       # Strategy comparisons & step-by-step deployment timeline
│       ├── simulator_view.py      # Battle playback controls, speed scaling, troop markers
│       ├── analytics_view.py      # Session statistics, OCR accuracy, dynamic CV latency
│       ├── dataset_view.py        # Screenshot labeling & bounding box editor
│       ├── settings_view.py       # Capture rates, OCR paths, farming thresholds
│       └── logs_view.py           # Structured event logging stream
└── core/                     # Common primitives
    ├── coordinates.py        # DPI scaling, relative-to-absolute coordinate transforms
    ├── models.py             # Strongly typed dataclasses
    └── config.py             # Persistent application configuration
```

---

## 3. Installation Requirements & Setup

### Prerequisites
- **Operating System**: Windows 11 (or Windows 10 64-bit)
- **Python**: Version 3.11 or higher (Python 3.13 supported)
- **Tesseract OCR**: Recommended for optical character recognition

### Step 1: Install Tesseract OCR on Windows
1. Download the Windows installer from [UB-Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
2. Install to the default directory: `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. Add `C:\Program Files\Tesseract-OCR` to your Windows system `PATH` environment variable.

### Step 2: Clone and Setup Python Environment
```powershell
# Open PowerShell in the project directory
cd "c:\Users\inte\Desktop\Cursor project\coc"

# Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Application Startup

Launch the desktop dashboard:
```powershell
python -m clashvision
```

---

## 5. Feature Guide

### 12-State Vision Recognition
The vision classifier automatically categorizes gameplay into 12 states:
- `HOME_VILLAGE`, `MULTIPLAYER_INTERFACE`, `MATCHMAKING_INTERFACE`
- `ENEMY_BASE_PREVIEW`, `BATTLE_IN_PROGRESS`, `BATTLE_RESULT`
- `TRAINING_INTERFACE`, `ARMY_CONFIGURATION`, `LOADING_SCREEN`
- `CONNECTION_ERROR`, `UNEXPECTED_POPUP`, `UNKNOWN`

Each classification returns confidence percentages, supporting evidence regions, and temporal debouncing to reject transition flickers.

### Numeric OCR Parsing
Extracts Gold, Elixir, Dark Elixir, troop quantities, and battle timers:
- Standard integers: `800,000` ➔ `800000`
- Metric abbreviations: `800K` ➔ `800000`, `1.2M` ➔ `1200000`
- European delimiters: `800.000` ➔ `800000`
- Timers: `2:45` ➔ `165` seconds
- Corrupt/ambiguous text safely returns `None` without guessing or hallucinating values.

### Farming Decision Engine
Bases are evaluated against user-configured resource thresholds:
- Minimum Gold, Elixir, Dark Elixir
- Resource priority weights: Gold, Elixir, Dark Elixir, or Balanced
- Layout scoring penalizes deep core storages and rewards exposed perimeter collectors
- Output provides an explainable score $[0.0, 100.0]$ with risk vs. reward metrics.

### 5 Attack Strategies
1. **Sneaky Goblin Farming**: Perimeter collector sweeps, jump/haste pathing, Town Hall snipes.
2. **BARCH**: Outer-ring building sweeps with alternating Barbarian and Archer deployment lines.
3. **Mass Miner**: Compartment navigation avoiding splash damage clusters.
4. **Mass Dragon**: Air defense corridor analysis and funneling corridors.
5. **Custom Strategy**: User-defined troop configurations, deployment priorities, and custom thresholds.

### Local Attack Simulator & Emergency Stop
- Provides full playback controls (`Play`, `Pause`, `Step Forward`, `Step Backward`, `Reset`).
- Variable playback speeds ($0.5\times$, $1.0\times$, $2.0\times$, $4.0\times$).
- Interactive troop deployment markers with visual decay lifetimes.
- Sandboxed controller strictly quarantined away from live game windows.
- Immediate emergency stop button halts playback in $< 0.2\text{ms}$.

---

## 6. Automated Testing

The project includes an offline, multi-tier test suite using synthetic and recorded fixtures. **No live game client or internet connection is required.**

Run the entire test suite:
```powershell
python tests/run_tests.py
```
Or directly with pytest:
```powershell
pytest -v tests/
```

### Test Suite Structure
- **Tier 1 (`tests/tiers/test_tier1_features.py`)**: Nominal acceptance criteria across all 31 requirements.
- **Tier 2 (`tests/tiers/test_tier2_boundaries.py`)**: Edge cases, window occlusions, zero-loot, corrupt OCR strings.
- **Tier 3 (`tests/tiers/test_tier3_pairwise.py`)**: Component interaction and pipeline transitions.
- **Tier 4 (`tests/tiers/test_tier4_workloads.py`)**: Full end-to-end simulated farming sessions.
- **Tier 5 (`tests/tiers/test_tier5_adversarial.py`)**: Chaos stress testing (1,000 emergency stops, 120 FPS frame floods, 4K/8K resolution scaling, concurrent SQLite WAL transactions).

---

## 7. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **Window Not Detected** | Google Play Games PC is not running or running as elevated Administrator. | Launch ClashVision as Administrator or start Google Play Games PC before attaching. |
| **Windows Display Scaling Issues** | DPI scaling causes coordinate offset. | The `WindowDetector` handles DPI automatically via Win32 `SetProcessDpiAwarenessContext`. Ensure display scaling is set between 100% and 200%. |
| **Missing Tesseract Executable** | Tesseract is not installed or not in system `PATH`. | Set the custom path in Settings View or install Tesseract to `C:\Program Files\Tesseract-OCR\tesseract.exe`. |
| **Incorrect OCR / Unknown Readings** | Glare or visual motion blur over resource numbers. | ClashVision's CLAHE contrast enhancement preprocessor automatically cleans noisy crops; uncertain readings are safely classified as unknown. |
| **Simulator Does Not Affect Live Game** | Safety isolation is working as designed. | Automated click injection is permanently blocked from live game windows by `SafetyGuard` and `SandboxedController`. |

---

## 8. Git Commit History

The repository is maintained locally with zero remote push, structured into atomic milestone commits:
- `ed18bcf`: `feat(infra): project architecture, test framework, and synthetic fixtures`
- `6be48f5`: `feat(capture): milestone 1 - safe window detection, coordinate scaling, and read-only guard`
- `e7b7a4e`: `feat(vision): milestone 2 - 12 game states classifier and numeric OCR parser`
- `7a1f4d3`: `feat(decision): milestone 3 - base evaluator and 5 attack strategy engines`
- `c41cf18`: `feat(simulator): milestone 4 - local attack replay simulator and emergency stop`
- `0cc08f5`: `feat(ui): milestone 5 - pyside6 dark dashboard with 9 views and sqlite wal persistence`
- `05a1a5e`: `test(e2e): milestone 6 - 100% e2e pass and tier 5 adversarial hardening`

---

## 9. Policy & Fair Play Notice

ClashVision is designed as an analytical study and assistance tool. Live gameplay decisions remain entirely in the hands of the user. In accordance with Supercell's Terms of Service and Safe and Fair Play Policy, ClashVision does not provide botting, memory editing, packet injection, or automated gameplay execution against live game servers.
