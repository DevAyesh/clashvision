# ClashVision — Research Report: Open-Source CoC Projects, Computer Vision, & Strategy Algorithms

**Document Version:** 1.0.0  
**Target Environment:** Windows 11 + Google Play Games PC (DirectX/Vulkan)  
**Safety Boundary:** Read-only Live Screen Observation; Isolated Local Simulator for All Autonomous Execution.  

---

## 1. Executive Summary

This research report investigates open-source Clash of Clans automation and computer vision projects, official documentation for core scientific and vision libraries (OpenCV, Tesseract OCR, RapidOCR, EasyOCR, MSS, PySide6, NumPy, Scikit-learn), and modern algorithms for:
- Resolution-independent screen capture and template matching
- High-precision numeric OCR and confidence scoring
- Base layout feature extraction, defense positioning, and collector exposure analysis
- Intelligent, explainable strategy selection and tactical battle simulation

---

## 2. Comparative Analysis of Public Repositories

| Repository | Tech Stack | Architecture & Approach | Strengths | Weaknesses | Relevance to ClashVision | License / Legal Boundary |
|---|---|---|---|---|---|---|
| **BasePilot** (`efebolukbasi/BasePilot`) | Python, PySide6, OpenCV, Tesseract OCR, PyInstaller | Desktop application targeting Google Play Games PC. Captures active game window, detects base elements via template matching, extracts gold/elixir via Tesseract, manages base upgrades and auto-rearming. | Native Windows 11 / Google Play Games PC support; clean PySide6 UI; robust error-handling for popups and connection drops; no memory injection. | Relies on fixed coordinate offsets for clicking; fragile to resolution changes; uses live-game click dispatch (bannable if automated live). | Window detection patterns, DPI awareness, popup handling logic, PySide6 widget styling. | MIT / Non-commercial; **Do NOT copy:** live click dispatch routines. |
| **coc-farming-bot** (`xsaintpy/coc-farming-bot`) | Python, OpenCV, Tesseract, ADB / LDPlayer | Headless/CLI emulator farming bot. Connects via ADB, cycles through matchmaking search, extracts loot via OCR, attacks based on threshold, deploys BARCH or Goblins across perimeter. | Mature loot threshold parsing; multi-pass OCR binarization; perimeter troop deployment coordinate math. | Tied to Android emulators via ADB; no native Windows client integration; primitive deployment scripting without base topology analysis. | Binarization thresholds for loot numbers, perimeter coordinate interpolation for troop drops. | GPLv3 / MIT; **Do NOT copy:** ADB tap commands, automated live matchmaking loops. |
| **COC-Auto-Attack** (`Ishanoshada/COC-Auto-Attack`) | Python, OpenCV, PyAutoGUI, NumPy | Script-based bot tailored for Google Play Games PC on Windows. Uses multi-scale template matching with OpenCV, lazy dependency loading, interactive calibration mode for screen resolution adjustments. | Multi-scale template matching (`cv2.matchTemplate` at varying scales); interactive calibration utility; lightweight memory footprint. | Uses PyAutoGUI direct mouse clicks to live game (high ban risk); lack of structured architecture or test suite; single-threaded blocking execution. | Multi-scale template matching algorithm for resolution invariance; coordinate letterboxing techniques. | MIT; **Do NOT copy:** PyAutoGUI live clicking actions. |
| **coc-bot** (`HeyFang/coc-bot`) | Python, OpenCV, NumPy | Modular bot framework for automated resource collection and troop training routines. | Clean separation of game state machine and image recognition steps. | Incomplete error recovery; lacks unit/integration testing; no base layout or defense topology awareness. | State transition design patterns for battle phases. | MIT; **Do NOT copy:** Live automated game input. |

### Strict Safety Boundary & Features That Must Not Be Copied:
1. **Live Input Dispatch:** Neither ADB taps, Win32 `SendMessage`/`PostMessage`, nor PyAutoGUI mouse clicks must ever be directed at Google Play Games or Clash of Clans processes.
2. **Anti-Detection & Memory Hooking:** No memory reading, DLL injection, API hooking, or behavioral jitter designed to bypass Supercell anti-cheat.
3. **Account Automation:** No gem spending, store purchasing, or automated live multiplayer matchmaking loops.

---

## 3. Computer Vision & Preprocessing Techniques

### A. Template Matching & Multi-Scale Invariance
In Google Play Games PC on Windows 11, users run at varying resolutions (720p, 1080p, 1440p, 4K) and DPI scalings (100%, 125%, 150%, 200%).
- **Standard `cv2.matchTemplate` Limitations:** Fails when the live window scale differs from the template scale by more than ~5%.
- **Multi-Scale Gaussian Pyramid Matching:**
  $$\text{Scale Range: } s \in [0.65, 1.35] \text{ with step } \Delta s = 0.05$$
  Resize input frame or template iteratively, execute `cv2.TM_CCOEFF_NORMED`, track global peak correlation `max_val` and bounding rect.
- **Edge-Based Matching (`cv2.Canny`):**
  Template matching over Canny edge maps is significantly more invariant to ambient lighting changes, grass tile color shifts (e.g., Winter/Halloween themes), and seasonal particle effects.

### B. Preprocessing Pipeline for Resource OCR
Clash of Clans numeric font (Supercell Magic / YouBlockhead) features white text with dark drop-shadows and dark brown/black outlines, rendered over semi-transparent grey dialogs:
1. **Color Space Isolation:**
   - Convert to Grayscale or isolate Value channel in HSV.
   - Text color is consistently near $(R, G, B) = (255, 255, 255)$. Mask high brightness pixels ($V \ge 220, S \le 35$).
2. **Contrast Enhancement (CLAHE):**
   - Apply Contrast Limited Adaptive Histogram Equalization with `clipLimit=2.5, tileGridSize=(4, 4)`.
3. **Otsu / Adaptive Thresholding:**
   - Invert binary image so characters are black on white background (optimal for Tesseract psm 7/8).
4. **Morphological Filtering:**
   - Apply morphological closing with a small $2 \times 2$ kernel to connect broken glyph stems caused by anti-aliasing artifacts.

---

## 4. OCR Engines: Comparative Analysis

| OCR Engine | Latency (CPU) | Accuracy on Game Numbers | Memory Footprint | Windows 11 / Offline Usability |
|---|---|---|---|---|
| **Tesseract OCR (v5.3+)** | 15–35 ms / crop | High with whitelist (`0-9,.:KMkm`) and PSM 7/8 | ~30 MB | Requires external native binary `tesseract.exe`; fast and deterministic when present. |
| **RapidOCR (ONNX Runtime)** | 20–45 ms / crop | Very High across distorted text | ~80 MB | Pure Python wheel + ONNX Runtime; zero native C++ installer needed; excellent standalone fallback. |
| **EasyOCR (PyTorch)** | 120–350 ms / crop | High | ~800 MB+ (PyTorch runtime) | Heavy weight; slow CPU inference; impractical for real-time 30+ FPS dashboards. |
| **MockOCR (ClashVision Native)** | < 0.1 ms | Deterministic synthetic | < 1 MB | Built-in zero-dependency engine for CI, test harnesses, and headless validation. |

### Confidence Evaluation & Sanity Filtering
- Parse confidence scores alongside extracted text.
- Rejection criteria:
  - Confidence $< 0.70$
  - Number parsing yields impossible values ($> 2,500,000$ per collector/storage pool for TH1-16)
  - Character ratio mismatch (presence of non-numeric glyphs when whitelist is applied)
- Fallback strategy: If OCR confidence is low, trigger manual review warning in UI and default to conservative risk assessment.

---

## 5. Base Layout & Defense Analysis

### A. Isometric Grid & Spatial Feature Extraction
Clash of Clans uses an isometric $44 \times 44$ tile map.
- **Center of Mass Calculation:**
  Identify base envelope by computing convex hull of all detected building contours.
- **Exposed Collector Ratio ($R_{exp}$):**
  $$R_{exp} = \frac{N_{\text{exposed}}}{N_{\text{total collectors}}}$$
  Collectors located in the outer 30% radial zone from the base centroid without enclosing wall contours are classified as exposed.
- **Town Hall Accessibility:**
  Radial distance from base centroid to Town Hall:
  - If $d_{\text{TH}} > 0.35 \times R_{\text{envelope}}$, Town Hall is off-center / snipable.
  - If $d_{\text{TH}} \le 0.35 \times R_{\text{envelope}}$ and enclosed by multi-layer walls, classified as deep core.
- **Air Defense Vectoring:**
  Detect Air Defense coordinates; compute convex polygon of AD coverage. Identify blind spots / corridors with zero overlapping air coverage for Dragon pathing.

---

## 6. Strategy Recommendation Intelligence

### Decision Matrix: Multi-Strategy Feasibility
```
                       [Base Layout & Resources Detected]
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
       Exposed Collectors ≥ 60%                      Air Defenses Clustered / Outer
                │                                             │
      ┌─────────┴─────────┐                                   │
      ▼                   ▼                                   ▼
[Sneaky Goblins]       [BARCH]                         [Mass Dragons]
(Snipe Storages/TH)  (Perimeter Sweep)               (Opposite-side Funnel)
```

### Explainable Scoring Formula
For each strategy $S \in \{\text{Sneaky Goblin, BARCH, Mass Dragon, Mass Miner, Custom}\}$:
$$\text{Score}(S) = w_r \cdot \text{LootScore} + w_e \cdot \text{Accessibility}(S) - w_d \cdot \text{DefenseHazard}(S) - w_c \cdot \text{TroopCostRatio}(S)$$
Every recommendation emits a structured rationale string explaining the decision factors (e.g., *"Selected Sneaky Goblin because 78% of collectors are outside wall perimeter and 850k Gold is snipable with low troop expenditure"*).

