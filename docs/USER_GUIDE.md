# ClashVision User Guide & Operating Manual

## 1. Introduction

**ClashVision** is a smart companion and autonomous battle simulator for Clash of Clans players on PC (Google Play Games PC on Windows 11).

ClashVision operates in two distinct modes:
1. **Live Observation Mode (Safe & Read-Only)**: Observes your live game screen, reads available loot in real-time, highlights high-value targets, and calculates recommended attack strategies.
2. **Autonomous Simulator Mode (Offline Sandbox)**: Runs automated simulated farming sessions in an isolated environment so you can test army compositions, optimize farming strategies, and benchmark yields without risking game penalties.

---

## 2. Getting Started

### 2.1 System Requirements
- Windows 11 (64-bit)
- Python 3.10+ (Tested on Python 3.13)
- Google Play Games PC (for live observation)
- Screen Resolution: 1080p (1920x1080) or 720p (1280x720)

### 2.2 Launching ClashVision
Launch the dashboard via:
```powershell
python -m clashvision.main
```
Or run in mock headless mode for automated testing:
```powershell
python -m clashvision.main --mock
```

---

## 3. Navigating the 10 Dashboard Views

| # | View Name | Purpose |
|---|-----------|---------|
| 1 | **Overview** | System health, active window connection, live loot summary, and fast metrics. |
| 2 | **Live Capture** | Real-time screen feed from Google Play Games PC with tactical overlays. |
| 3 | **Base Analyzer** | Spatial layout breakdown: exposed collectors vs core storages, symmetry, difficulty, and hazard entry points. |
| 4 | **Strategy Planner** | Multi-strategy comparative ranking across all 5 archetypes with economic ROI projections. |
| 5 | **Simulator** | Interactive step-by-step playback of procedural attack plans with speed controls. |
| 6 | **Analytics** | Historical farming performance, total resources harvested, and hourly gain graphs. |
| 7 | **Dataset** | Dataset management, visual annotations, and bounding box inspection. |
| 8 | **Settings** | Minimum loot thresholds, OCR backend selection, and capture frame rate. |
| 9 | **Logs** | Real-time diagnostic logs and safety audit trail. |
| 10 | **Autonomous Simulator** | End-to-end autonomous farming cycle orchestrator in the local sandbox. |

---

## 4. Using the Autonomous Farming Simulator

1. Navigate to **Autonomous Simulator** (View 10).
2. Configure your desired target criteria:
   - **Filter Mode**:
     - `ALL`: Both Gold, Elixir, and Dark Elixir must meet thresholds.
     - `ANY`: Target accepted if any resource exceeds threshold.
     - `WEIGHTED`: Target accepted based on total economic value.
     - `CUSTOM`: Evaluates custom tactical conditions.
   - **Min Gold / Elixir / DE**: Minimum values required before initiating a simulated attack.
   - **Max Search Attempts**: Maximum base skips allowed per cycle.
   - **Max Cycles**: Number of continuous farming loops (e.g. 10).
3. Click **▶ Start Simulation**.
4. The engine will automatically:
   - Scout procedural candidate bases.
   - OCR and evaluate loot amounts.
   - Skip unsuitable targets and select viable candidates.
   - Select the optimal strategy (Sneaky Goblins, BARCH, Miners, Dragons).
   - Simulate deployment, troop interactions, and defense destruction.
   - Resolve stars and loot collected.
   - Record results to database and reset for the next cycle.
5. Click **🛑 EMERGENCY STOP** at any time to immediately freeze simulation in under 50ms.

---

## 5. Exporting Farming Analytics

Export your historical session data at any time:
- In **Autonomous Simulator**, click **📊 Export Session Analytics**.
- Or via Python API:
```python
from clashvision.storage.repository import Repository
repo = Repository()

# Export session data to JSON
repo.export_sessions_to_json("sessions_export.json")

# Export all base scans to CSV
repo.export_scans_to_csv("base_scans.csv")

# Export battle simulation results
repo.export_battles_to_json("battles.json")
```

