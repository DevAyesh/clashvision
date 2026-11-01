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
│  - Commit e45f90f | Baseline verified: 435 tests passing [COMPLETED]   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│             PHASE 2: MULTI-SCALE TEMPLATE MATCHING ENGINE              │
│  - Resolution-independent matching (720p, 1080p, 1440p, 4K)            │
│  - Configurable scale space with early exit optimization               │
│  - Grayscale & Canny edge-based matching                               │
│  - Commit 25a8a0f | Verified: 447 tests passing (+12 tests) [COMPLETED]│
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│               PHASE 3: READ-ONLY SCREEN ANALYSIS ENGINE                │
│  - Multi-scale template matching (resolution invariance)               │
│  - OCR normalization & optional EasyOCR backend                        │
│  - Preprocessing and confidence scoring                                │
│  - Commits: bba8c0a, 967b20f, 50904be, 78b825a [COMPLETED]             │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                  PHASE 4: LOOT EVALUATION ENGINE                       │
│  - Configurable multi-mode evaluation (Individual, Combined, Weighted) │
│  - Confidence-weighted score degradation for noisy OCR                 │
│  - Structured explainable decision audit trails                        │
│  - Implemented & verified within decision & model pipelines [COMPLETED]│
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                   PHASE 5: BASE ANALYSIS ENGINE                        │
│  - Base symmetry & centroid calculation                                │
│  - Core vs perimeter resource distribution & accessibility             │
│  - Defense coverage hazard & entry point sector rating                 │
│  - Commits: 2f3bff0, 12130d2, 312adfd, ce408b0 [COMPLETED]             │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│              PHASE 6: STRATEGY RECOMMENDATION ENGINE                   │
│  - Weighted strategy ranking across 5 distinct archetypes              │
│  - Economic ROI projections (army training costs vs projected net gain)│
│  - Explainable pros/cons rationales and recommended entry sectors      │
│  - Commits: 21e35e8, 1b511a6 [COMPLETED]                               │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│              PHASE 7: AUTONOMOUS LOCAL SIMULATOR                       │
│  - Synthetic procedural base generator (5 topology archetypes)         │
│  - Autonomous target selection, consecutive skips & streak tracking    │
│  - Isolated deployment execution & analytical battle resolution        │
│  - Error recovery, listeners, and farming loop controls                │
│  - Commits: 94b2bcc, 4cebd3e, bb0deae, fae45bd [COMPLETED]             │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│              PHASE 8: DASHBOARD & UI INTEGRATION                       │
│  - Visual topology badges and economic ROI indicators in base views    │
│  - Autonomous simulator dashboard: streak counters, export triggers    │
│  - Commits: 901bfad, 93028bc [COMPLETED]                               │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                 PHASE 9: DATA & SESSION ANALYTICS                      │
│  - JSON & CSV export utilities for sessions, scans, and battles        │
│  - Centralized resource manager for cached assets & memory bounds      │
│  - Commits: 2a08baa, f73aadb [COMPLETED]                               │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                  PHASE 10: VERIFICATION & TESTING                      │
│  - Comprehensive end-to-end simulation & safety verification suite     │
│  - Strict isolation: live window injection blocked; safety trip <0.1ms │
│  - Complete test baseline: 480 passed, 125 subtests passed [COMPLETED] │
│  - Commits: 5d4bbab, (docs release) [COMPLETED]                        │
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

