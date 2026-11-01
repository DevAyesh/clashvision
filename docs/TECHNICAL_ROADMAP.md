# ClashVision — Technical Roadmap & Implementation Plan

**Branch:** `clashvision-research-improvements`  
**Target Environment:** Windows 11 + Google Play Games PC (DirectX/Vulkan)  
**Safety Protocol:** Strictly Read-Only Live Screen Observation; Quarantined Local Simulator for Autonomous Execution.  

---

## 1. Roadmap Architecture & Phases

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: RESEARCH & AUDIT                       │
│  - Survey open-source CoC bots (BasePilot, coc-farming-bot, etc.)     │
│  - Fix UI test baseline (10 views, mock widgets, singleton reset)      │
│  - Baseline verified: 435 tests passing [COMPLETED]                    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│             PHASE 2: MULTI-SCALE TEMPLATE MATCHING ENGINE              │
│  - Resolution-independent matching (720p, 1080p, 1440p, 4K)            │
│  - Configurable scale space with early exit optimization               │
│  - Grayscale & Canny edge-based matching                               │
│  - Benchmarks: Exhaustive 429ms (85ms/scale), Early-exit 248ms         │
│  - Verified: 447 tests passing (+12 new tests) [COMPLETED]             │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│               PHASE 3: READ-ONLY SCREEN ANALYSIS ENGINE                │
│  - Multi-scale template matching (resolution invariance)               │
│  - Configurable capture regions (ROI scaling for 720p/1080p/1440p/4K)  │
│  - Optional EasyOCR backend in pluggable OCR engine                    │
│  - Image dataset and offline fixture loader                            │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                  PHASE 4: LOOT EVALUATION ENGINE                       │
│  - Configurable multi-mode evaluation (Individual, Combined, Weighted)  │
│  - Confidence-weighted score degradation for noisy OCR                 │
│  - Structured explainable decision audit trails                        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                   PHASE 5: BASE ANALYSIS ENGINE                        │
│  - Base symmetry and centroid calculation                              │
│  - Resource concentration heatmapping                                  │
│  - Defense coverage zones and entry point candidate extraction         │
│  - Tactical difficulty rating [1–10]                                   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│              PHASE 6: STRATEGY RECOMMENDATION ENGINE                   │
│  - Multi-strategy profile definitions with troop cost & damage models  │
│  - Success probability calculation based on base topology             │
│  - Structured explainability strings for every recommendation          │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│              PHASE 7: AUTONOMOUS LOCAL SIMULATOR                       │
│  - Synthetic base generator with customizable layouts & defenses       │
│  - Automatic target selection/skipping loops                           │
│  - Action executor with pause, resume, and emergency stop controls     │
│  - Battle outcome and resource acquisition calculations                │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│              PHASE 8: DASHBOARD & UI INTEGRATION                       │
│  - Align all 10 views in PySide6 dashboard                             │
│  - Real-time simulation counters, gauges, and log feeds                │
│  - Fix all 9 headless UI integration tests                             │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                 PHASE 9: DATA & SESSION ANALYTICS                      │
│  - Persistent SQLite WAL storage schema updates                        │
│  - Export sessions and attack analytics to JSON & CSV                  │
│  - Aggregated performance metrics (Loot/Hour, Win Rates)               │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                  PHASE 10: VERIFICATION & TESTING                      │
│  - Execute full unit, tier-boundary, adversarial, and integration tests │
│  - Verify 100% pass rate across entire test suite                      │
│  - Verify emergency stop latency (< 0.25 ms)                           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Implementation Phases

### Phase 3: Read-Only Screen Analysis Engine
- **Files to Modify/Extend:**
  - `clashvision/vision/ocr_engine.py`: Add `EasyOCREngine` support.
  - `clashvision/vision/preprocessor.py`: Add multi-scale template matching helper `match_template_multiscale()`.
  - `clashvision/capture/window_detector.py`: Refine DPI scaling calculation.
- **Verification:** Unit tests for multi-scale template matching and OCR confidence thresholds.

### Phase 4: Configurable Loot Evaluation Engine
- **Files to Modify/Extend:**
  - `clashvision/decision/evaluator.py`: Add combined loot score formulas and confidence-weighted discounting.
  - `clashvision/simulator/target_selector.py`: Verify filtering modes (`ALL`, `ANY`, `WEIGHTED`, `CUSTOM`).
- **Verification:** Test cases covering edge cases (zero values, OCR corruption, extreme numbers).

### Phase 5: Base Analysis Engine
- **Files to Modify/Extend:**
  - `clashvision/vision/layout_analyzer.py`: Add symmetry estimation, entry point candidates, and tactical difficulty scoring.
- **Verification:** Tests on synthetic base layouts with varying wall structures and collector spreads.

### Phase 6: Strategy Recommendation Engine
- **Files to Modify/Extend:**
  - `clashvision/decision/recommendation_engine.py` & strategy modules: Add troop cost calculation, damage projections, and detailed rationales.
- **Verification:** Strategy eligibility and ranking tests across varied army compositions.

### Phase 7: Autonomous Local Simulator
- **Files to Modify/Extend:**
  - `clashvision/simulator/autonomous_engine.py`
  - `clashvision/simulator/battle_resolver.py`
  - `clashvision/simulator/deployment_executor.py`
- **Verification:** End-to-end multi-cycle simulation tests.

### Phase 8: Dashboard & UI Harmonization
- **Files to Modify/Extend:**
  - `clashvision/ui/main_window.py`
  - `clashvision/ui/views/autonomous_simulator_view.py`
  - Fix test suite expectations in `tests/integration/test_ui_headless.py` and `tests/unit/test_adversarial_m5_ui.py`.
- **Verification:** All 10 views switch cleanly and all headless UI tests pass.

### Phase 9: Data & Session Analytics
- **Files to Modify/Extend:**
  - `clashvision/storage/repository.py`: Add CSV and JSON export methods for sessions, base scans, and battle results.
- **Verification:** Tests validating database export format and integrity.

### Phase 10: Verification & Testing
- **Execution:** Run pytest over all unit, tier, integration, and adversarial suites. Ensure zero regressions.

