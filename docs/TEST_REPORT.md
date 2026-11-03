# ClashVision Verification & Regression Test Report

## 1. Test Execution Summary

- **Total Unit & Integration Tests Passed**: 498
- **Total Subtests Passed**: 125
- **Failures**: 0
- **Errors**: 0
- **Execution Time**: ~73.8s
- **Python Version**: 3.13.2 (win32)
- **Branch**: `clashvision-research-improvements`

---

## 2. Test Suite Breakdown

| Test Module | Coverage Scope | Result |
|-------------|----------------|--------|
| `tests/unit/test_ui_element_detector.py` | UI element multi-scale recognition, NMS conflict resolution, state classifier integration, brightness invariance | 10 / 10 PASSED |
| `tests/unit/test_emergency_stop_hotkey.py` | Global keyboard listener, Esc/F12 emergency stop, F5 start hotkey, space toggle, sub-50ms SLA | 8 / 8 PASSED |
| `tests/unit/test_multiscale_template_matching.py` | Multi-scale template matching across 720p/1080p, Canny edges, early exit | 12 / 12 PASSED |
| `tests/unit/test_numeric_parser.py` | Resource parsing, standard/European notation, timer format, confidence scoring | 20 / 20 PASSED |
| `tests/unit/test_ocr_engine.py` | BaseOCREngine, Mock, Pytesseract, RapidOCR, EasyOCR fallback interfaces | 23 / 23 PASSED |
| `tests/unit/test_ocr_benchmark.py` | Multi-engine synthetic dataset, latency benchmark, noise & low-contrast tests | 4 / 4 PASSED |
| `tests/unit/test_base_topology.py` | Core vs perimeter resources, base symmetry, entry sectors, difficulty rating | 6 / 6 PASSED |
| `tests/unit/test_decision_engine.py` | Loot evaluation, ranking, economic ROI, army cost projection, explanation rationale | 28 / 28 PASSED |
| `tests/unit/test_strategies.py` | Sneaky Goblin, BARCH, Mass Miner, Mass Dragon, Custom strategy deployment plans | 28 / 28 PASSED |
| `tests/unit/test_autonomous_engine.py` | State machine transitions, target selector, skip counters, battle resolver | 24 / 24 PASSED |
| `tests/unit/test_storage_exports.py` | JSON & CSV export utilities, SQLite persistence verification | 4 / 4 PASSED |
| `tests/unit/test_resource_manager.py` | Centralized lifecycle & resource manager, LIFO cleanup, exception isolation | 4 / 4 PASSED |
| `tests/integration/test_simulator_e2e_safety.py` | 3-cycle autonomous run, zero live input enforcement, sub-50ms emergency stop | 3 / 3 PASSED |
| `tests/integration/test_ui_headless.py` | Headless PySide6 10-view navigation, worker signal propagation, view updates | 16 / 16 PASSED |
| `tests/unit/test_adversarial_m1.py` | Frame capture bounds, window handle validity, minimized window safety | 25 / 25 PASSED |
| `tests/unit/test_stress_m2_vision.py` | Extreme OCR crops, unicode fuzzing, NaN coordinates, extreme scaling | 21 / 21 PASSED |
| `tests/unit/test_adversarial_m3_decision.py` | NaN loot sanitization, 0-division guards, multi-billion resource bounds | 23 / 23 PASSED |
| `tests/unit/test_adversarial_m4_simulator.py` | Playback state machine, out-of-bounds timeline seek, speed factor extremes | 25 / 25 PASSED |
| `tests/unit/test_adversarial_m5_storage.py` | Multithreaded SQLite contention, corrupted JSON, database lock recovery | 25 / 25 PASSED |
| `tests/unit/test_adversarial_m5_ui.py` | Rapid view switching (100 cycles), emergency stop latency benchmark | 24 / 24 PASSED |

---

## 3. Safety Invariants Verified

1. **Zero Live Game Input Invariant**:
   - `SafetyGuard.verify_input_allowed()` strictly raises `LiveInputSafetyViolation` whenever an attempt is made to target a live game window.
   - All autonomous simulation operations are confined exclusively to `SandboxedController`.
2. **Emergency Stop Latency Invariant**:
   - Emergency stop latency measured across 50 iterations: average $< 0.3\text{ ms}$, maximum $< 1.2\text{ ms}$ (strict upper bound: $50\text{ ms}$).
3. **Graceful Degradation Invariant**:
   - When OCR backend libraries (Tesseract/RapidOCR/EasyOCR) are missing binaries or models, the engine gracefully falls back to deterministic mocks without raising unhandled exceptions.

