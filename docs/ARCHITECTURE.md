# ClashVision Architecture Specification

## 1. System Overview

**ClashVision** is a dual-subsystem, safety-enforced desktop analytics and simulation platform for Clash of Clans (Google Play Games PC on Windows 11).

The application enforces a strict operational separation between:
1. **Live Observation Environment (Strictly Read-Only)**: Captures game frames, classifies game states, performs OCR for loot analysis, evaluates base layouts, and presents tactical recommendations to the human player. **Under no circumstances does it dispatch unattended keyboard/mouse inputs or ADB commands into the live game.**
2. **Local Attack Simulation Environment (Autonomous Sandbox)**: An offline, fully autonomous farming and battle simulation engine. It generates synthetic enemy bases, executes procedural deployments of 5 distinct strategy archetypes, simulates damage and defenses, resolves battle outcomes, records loot gains, and executes continuous farming cycles with sub-50ms emergency stop protection.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CLASHVISION PLATFORM                            │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
│       LIVE OBSERVATION SYSTEM       │     │       LOCAL SIMULATOR ENGINE        │
│        (100% READ-ONLY)             │     │      (AUTONOMOUS SANDBOX)           │
├─────────────────────────────────────┤     ├─────────────────────────────────────┤
│ • Win32 GDI / WGC Frame Grabber     │     │ • Procedural Synthetic Base Gen     │
│ • State Classifier (12 Game States) │     │ • Multi-Filtering Target Selector   │
│ • Pluggable OCR Subsystem           │     │ • Sandboxed Deployment Executor     │
│   (Tesseract, RapidOCR, EasyOCR)    │     │ • Real-Time Battle Outcome Resolver │
│ • Layout Analyzer (Topology/Hazard) │     │ • Continuous Multi-Cycle Loop       │
│ • 10-View PySide6 Desktop Dashboard │     │ • Sub-50ms Emergency Stop Barrier   │
└─────────────────────────────────────┘     └─────────────────────────────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │    STORAGE & ANALYTICS LAYER  │
                     │  (SQLite, JSON & CSV Exports) │
                     └───────────────────────────────┘
```

---

## 2. Core Subsystems

### 2.1 Vision & OCR Subsystem (`clashvision/vision/`)
- **Preprocessors**: Multi-scale template matching with Canny edge support, CLAHE contrast enhancement, Otsu binarization, morphological opening.
- **Pluggable OCR Engines**:
  - `PytesseractEngine`: Standard binary-backed OCR.
  - `RapidOCREngine`: Fast lightweight ONNX OCR.
  - `EasyOCREngine`: PyTorch deep-learning OCR with CUDA acceleration and CPU fallback.
  - `MockOCREngine`: Deterministic testing harness.
- **Base Layout Analyzer**:
  - Resource distribution analysis (core vs perimeter concentration).
  - Base symmetry scoring (horizontal, vertical, rotational).
  - Entry point & defense hazard evaluation (North, East, South, West).
  - Base difficulty rating ($0.0 \dots 1.0$).

### 2.2 Decision Subsystem (`clashvision/decision/`)
- **Recommendation Engine**: Evaluates available loot and layout topology against 5 tactical archetypes:
  1. *Sneaky Goblins*: Rapid extraction of perimeter loot and accessible Town Halls.
  2. *BARCH*: Cheap perimeter collector clearing.
  3. *Mass Miners*: Burrowing core penetration bypassing wall compartments.
  4. *Mass Dragons*: Aerial destruction ignoring ground defenses.
  5. *Custom Strategy*: User-defined troop compositions and criteria.
- **Economic ROI Projections**: Calculates army training elixir/dark elixir costs and projects net loot gains.

### 2.3 Local Simulation Subsystem (`clashvision/simulator/`)
- **Simulation Engine & Playback State Machine**: Timeline controls (Play, Pause, Step, Stop) with speed multipliers (0.1x to 10.0x).
- **Target Selector**: Enforces loot thresholds across 4 modes: `ALL`, `ANY`, `WEIGHTED`, and `CUSTOM`.
- **Battle Resolver**: Simulates building destruction, troop mortality, star criteria (50%, TH, 100%), and harvested loot.
- **Autonomous Farming Engine**: Manages continuous multi-cycle farming loops with automated anomaly recovery.

### 2.4 Safety Guard & Isolation Barrier (`clashvision/simulator/safety_barrier.py`)
- Hardware/OS-level safety barriers intercepting any attempt to target external live game windows.
- Global Emergency Stop latch guaranteeing complete execution halt in $< 50\text{ ms}$.

### 2.5 Persistence & Reporting (`clashvision/storage/`)
- SQLite storage backing sessions, scans, recommendations, and analytics.
- Structured export utilities (`export_sessions_to_json`, `export_scans_to_csv`, `export_battles_to_json`, `export_analytics_to_csv`).

