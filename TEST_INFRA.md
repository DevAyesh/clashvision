# E2E Test Infra: ClashVision

## Test Philosophy
- **Requirement-Driven & Opaque-Box**: Derived directly from `ORIGINAL_REQUEST.md` (R1–R6), not internal implementation nuances.
- **100% Offline Execution**: Zero live connection to Google Play Games PC required. All tests execute against programmatic synthetic fixtures and deterministic recorded ground truths.
- **Strict Safety Interception**: Global autouse pytest fixture intercepts and blocks any call to live mouse/keyboard APIs (`pyautogui`, Win32 `SendInput`, `mouse_event`), raising `LiveInputSafetyViolation` if detected outside quarantined simulator bounds.
- **Methodology**: Systematic 4-tier matrix (Category-Partition, Boundary Value Analysis, Pairwise Combinatorial, Real-World Workload Testing).

---

## Feature Inventory & Test Mapping

| # | Feature | Requirement | Tier 1 (Nominal) | Tier 2 (Boundary) | Tier 3 (Pairwise) | Tier 4 (Workload) |
|---|---------|-------------|:----------------:|:-----------------:|:-----------------:|:-----------------:|
| 1 | Window Enumeration & Target Discovery | R1 | 5 | 5 | ✓ | ✓ |
| 2 | High-Performance MSS Frame Capture | R1 | 5 | 5 | ✓ | ✓ |
| 3 | DPI Scaling & Dynamic Resize Transform | R1 | 5 | 5 | ✓ | ✓ |
| 4 | Window State & Visibility Monitor | R1 | 5 | 5 | ✓ | ✓ |
| 5 | Read-Only Live Safety Guard | R1, R4 | 5 | 5 | ✓ | ✓ |
| 6 | 12 Game State Classifier | R2 | 12 | 12 | ✓ | ✓ |
| 7 | State Classifier Evidence & Confidence | R2 | 5 | 5 | ✓ | ✓ |
| 8 | Multi-Frame State Debouncing | R2 | 5 | 5 | ✓ | ✓ |
| 9 | Resource ROI Localization | R2 | 5 | 5 | ✓ | ✓ |
| 10 | Vision Image Preprocessing | R2 | 5 | 5 | ✓ | ✓ |
| 11 | Pluggable OCR Engine Abstraction | R2 | 5 | 5 | ✓ | ✓ |
| 12 | Numeric OCR Sanitizer & Value Parser | R2 | 10 | 10 | ✓ | ✓ |
| 13 | Visual OCR Debug Data Model | R2 | 5 | 5 | ✓ | ✓ |
| 14 | Configurable Farming Base Evaluator | R3 | 5 | 5 | ✓ | ✓ |
| 15 | Base Layout Accessibility Analysis | R3 | 5 | 5 | ✓ | ✓ |
| 16 | Strategy A: Sneaky Goblin Farming | R3 | 5 | 5 | ✓ | ✓ |
| 17 | Strategy B: BARCH Farming | R3 | 5 | 5 | ✓ | ✓ |
| 18 | Strategy C: Mass Miner | R3 | 5 | 5 | ✓ | ✓ |
| 19 | Strategy D: Mass Dragon | R3 | 5 | 5 | ✓ | ✓ |
| 20 | Strategy E: Custom User Strategy | R3 | 5 | 5 | ✓ | ✓ |
| 21 | Explainable Recommendation Breakdown | R3 | 5 | 5 | ✓ | ✓ |
| 22 | Local Attack Simulator Engine | R4 | 5 | 5 | ✓ | ✓ |
| 23 | Troop Deployment & Path Simulation | R4 | 5 | 5 | ✓ | ✓ |
| 24 | Simulator Playback Controls | R4 | 5 | 5 | ✓ | ✓ |
| 25 | Simulator Sandboxed Input Isolation | R4 | 5 | 5 | ✓ | ✓ |
| 26 | Emergency Stop Safety Control | R4 | 5 | 5 | ✓ | ✓ |
| 27 | Visual Mode Safety Indicator | R4 | 5 | 5 | ✓ | ✓ |
| 28 | Dark-Themed PySide6 Dashboard Shell | R5 | 5 | 5 | ✓ | ✓ |
| 29 | 9 Functional Sidebar Views | R5 | 9 | 9 | ✓ | ✓ |
| 30 | SQLite WAL Persistence Layer | R5 | 5 | 5 | ✓ | ✓ |
| 31 | Non-Blocking QThread Worker Pool | R5 | 5 | 5 | ✓ | ✓ |

---

## Test Architecture

### Runners and Execution Commands
1. **Primary Pytest Runner**:
   ```powershell
   python -m pytest tests/ -v --durations=10
   ```
2. **Tier-Specific Execution**:
   ```powershell
   python -m pytest tests/tiers/test_tier1_features.py -v
   python -m pytest tests/tiers/test_tier2_boundaries.py -v
   python -m pytest tests/tiers/test_tier3_pairwise.py -v
   python -m pytest tests/tiers/test_tier4_workloads.py -v
   ```
3. **Zero-Dependency Fallback Runner**:
   ```powershell
   python tests/run_tests.py
   ```

### Pass / Fail Semantics
- Exit code `0`: All tests pass with zero assertions failed and zero safety violations.
- Exit code `!= 0`: Any test failure, unhandled exception, or safety violation.
- Any live input invocation outside simulator bounds raises `LiveInputSafetyViolation` and fails the suite immediately.

### Directory Layout
```
tests/
├── conftest.py                   # Pytest fixtures, mock OCR, safety interceptor
├── run_tests.py                  # Self-contained test runner
├── fixtures/
│   ├── synthetic_generator.py    # Generates 12 states, OCR variations & base layouts
│   ├── layout_generator.py       # Graph structures for 6 canonical base configurations
│   └── recorded_samples.py       # Synthetic and recorded ground truth samples
├── unit/                         # Fast unit tests for components
├── integration/                  # Pipeline integration (Capture -> Vision -> Decision)
└── tiers/                        # 4-tier requirement validation
    ├── test_tier1_features.py    # >= 5 test cases per feature (nominal)
    ├── test_tier2_boundaries.py  # >= 5 test cases per feature (stress & edge)
    ├── test_tier3_pairwise.py    # Pairwise subsystem interactions
    └── test_tier4_workloads.py   # Realistic end-to-end farming workflows
```

---

## Test Coverage Thresholds

- **Tier 1 (Feature Coverage)**: At least 150 nominal test cases verifying every inventoried feature in isolation.
- **Tier 2 (Boundary & Corner Cases)**: At least 150 boundary test cases testing limits, zero/negative resources, malformed strings ("800M", "???", "---"), corrupted crops, minimized windows, rapid emergency stops.
- **Tier 3 (Cross-Feature Pairwise)**: At least 20 test cases testing state transitions + OCR updates + decision re-computation + simulator handoff.
- **Tier 4 (Real-World Application Workloads)**: At least 5 full-session simulated scenarios (Scout -> Next Base -> High Loot Target -> Recommendation -> Simulate Attack -> Results).
- **Total Suite Minimum**: > 325 test cases.
