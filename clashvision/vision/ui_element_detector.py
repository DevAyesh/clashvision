"""
clashvision/vision/ui_element_detector.py

Resolution-Independent Visual UI Element Recognition and Classification Module.
Detects attack buttons, matchmaking controls, confirmation dialogs, and navigation buttons
across arbitrary resolutions, scaling levels, and lighting variations using multi-scale OpenCV matching.
Maintains strict read-only observation guarantees.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from clashvision.core.constants import (
    GameState,
    VIRTUAL_REFERENCE_WIDTH,
    VIRTUAL_REFERENCE_HEIGHT,
)
from clashvision.core.models import (
    Point,
    NormalizedPoint,
    Rect,
    NormalizedRect,
)
from clashvision.vision.preprocessor import VisionPreprocessor, TemplateMatchResult

logger = logging.getLogger("clashvision.vision.ui_element_detector")


@unique
class UIElementType(str, Enum):
    """Recognized visual UI element types."""
    BATTLE_BUTTON = "battle_button"                   # "Battle" button at Home Village (bottom-left)
    FIND_MATCH_BUTTON = "find_match_button"           # "Find a Match" (1,500 Gold) button
    ATTACK_CONFIRM_BUTTON = "attack_confirm_button"   # Green "Attack!" button on Army overview modal
    NEXT_BUTTON = "next_button"                       # "Next" / Skip candidate button (bottom-right)
    END_BATTLE_BUTTON = "end_battle_button"           # "End Battle" / Surrender button (bottom-left)
    RETURN_HOME_BUTTON = "return_home_button"         # "Return Home" / Okay button (post-battle)
    CLOSE_MODAL_BUTTON = "close_modal_button"         # Red 'X' modal close button
    EDIT_ARMY_BUTTON = "edit_army_button"             # "Edit Army" button on matchmaking screen
    SURRENDER_CONFIRM_BUTTON = "surrender_confirm_button"  # Dialog confirmation button


# Expected state association mapping
ELEMENT_STATE_MAP: Dict[UIElementType, GameState] = {
    UIElementType.BATTLE_BUTTON: GameState.HOME_VILLAGE,
    UIElementType.FIND_MATCH_BUTTON: GameState.MULTIPLAYER_INTERFACE,
    UIElementType.ATTACK_CONFIRM_BUTTON: GameState.ARMY_CONFIGURATION,
    UIElementType.NEXT_BUTTON: GameState.ENEMY_BASE_PREVIEW,
    UIElementType.END_BATTLE_BUTTON: GameState.BATTLE_IN_PROGRESS,
    UIElementType.RETURN_HOME_BUTTON: GameState.BATTLE_RESULT,
    UIElementType.EDIT_ARMY_BUTTON: GameState.MULTIPLAYER_INTERFACE,
}

# Color palette for bounding box visualization
ELEMENT_COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    UIElementType.BATTLE_BUTTON.value: (0, 215, 255),           # Gold (BGR)
    UIElementType.FIND_MATCH_BUTTON.value: (0, 165, 255),       # Orange
    UIElementType.ATTACK_CONFIRM_BUTTON.value: (46, 204, 113),   # Green
    UIElementType.NEXT_BUTTON.value: (30, 144, 255),            # Deep Orange
    UIElementType.END_BATTLE_BUTTON.value: (50, 50, 220),       # Red
    UIElementType.RETURN_HOME_BUTTON.value: (0, 200, 255),      # Gold
    UIElementType.CLOSE_MODAL_BUTTON.value: (100, 100, 255),    # Light Red
    UIElementType.EDIT_ARMY_BUTTON.value: (255, 180, 50),       # Cyan-Blue
    UIElementType.SURRENDER_CONFIRM_BUTTON.value: (60, 60, 230),# Red
}


@dataclass
class UIElementResult:
    """Detection result for an individual visual UI element."""
    element_type: str
    bounding_box: Rect
    normalized_box: NormalizedRect
    center_point: Point
    normalized_center: NormalizedPoint
    confidence: float
    scale: float
    state_association: Optional[GameState] = None
    template_name: str = ""
    is_ambiguous: bool = False
    alternative_matches: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_type": self.element_type,
            "bounding_box": {
                "x": self.bounding_box.x,
                "y": self.bounding_box.y,
                "width": self.bounding_box.width,
                "height": self.bounding_box.height,
            },
            "normalized_box": {
                "x": round(self.normalized_box.x, 4),
                "y": round(self.normalized_box.y, 4),
                "width": round(self.normalized_box.width, 4),
                "height": round(self.normalized_box.height, 4),
            },
            "center": (self.center_point.x, self.center_point.y),
            "normalized_center": (round(self.normalized_center.x, 4), round(self.normalized_center.y, 4)),
            "confidence": round(self.confidence, 4),
            "scale": round(self.scale, 3),
            "state_association": self.state_association.value if self.state_association else None,
            "template_name": self.template_name,
            "is_ambiguous": self.is_ambiguous,
            "metadata": self.metadata,
        }


@dataclass
class ElementRecognitionConfig:
    """Configuration parameters for UI element detection and classification."""
    scales: List[float] = field(default_factory=lambda: [0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 1.875, 2.0])
    min_confidence: float = 0.65
    high_confidence: float = 0.85
    early_exit_threshold: float = 0.95
    nms_iou_threshold: float = 0.35
    ambiguity_margin: float = 0.08
    method: str = "grayscale"  # "grayscale" or "edge"
    color_verification: bool = True
    save_debug_annotated_images: bool = False
    debug_output_dir: str = "debug_detections"


@dataclass
class DetectionDiagnostics:
    """Detailed telemetry and audit diagnostics for UI element recognition."""
    total_detections: int = 0
    execution_time_ms: float = 0.0
    elements_found: List[str] = field(default_factory=list)
    ambiguous_count: int = 0
    conflict_count: int = 0
    frame_resolution: Tuple[int, int] = (0, 0)
    associated_state: Optional[GameState] = None
    scores_by_element: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_detections": self.total_detections,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "elements_found": self.elements_found,
            "ambiguous_count": self.ambiguous_count,
            "conflict_count": self.conflict_count,
            "frame_resolution": self.frame_resolution,
            "associated_state": self.associated_state.value if self.associated_state else None,
            "scores_by_element": {k: round(v, 4) for k, v in self.scores_by_element.items()},
            "details": self.details,
        }


class UIElementDetector:
    """
    Resolution-independent multi-scale UI element recognition engine.
    Locates and classifies interactive buttons, modals, and navigation controls
    using multi-template OpenCV correlation and color verification.
    """

    def __init__(
        self,
        config: Optional[ElementRecognitionConfig] = None,
        template_dir: Optional[str] = None,
    ) -> None:
        self.config = config or ElementRecognitionConfig()
        self.template_dir = template_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "templates"
        )
        self._templates: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        self._last_diagnostics: Optional[DetectionDiagnostics] = None

        # Load file-based and procedural fallback templates
        self._load_templates()

    def _load_templates(self) -> None:
        """Loads reference templates from disk and registers synthetic fallbacks."""
        # Check assets folder
        if os.path.isdir(self.template_dir):
            template_mappings = {
                UIElementType.BATTLE_BUTTON.value: ["battle_button.png", "battle.png", "btn_battle.png"],
                UIElementType.FIND_MATCH_BUTTON.value: ["find_match_button.png", "find_match.png", "btn_find_match.png"],
                UIElementType.ATTACK_CONFIRM_BUTTON.value: ["attack_confirm_button.png", "attack_button.png", "btn_attack.png"],
                UIElementType.NEXT_BUTTON.value: ["next_button.png", "next.png", "btn_next.png"],
                UIElementType.END_BATTLE_BUTTON.value: ["end_battle_button.png", "end_battle.png", "surrender.png"],
                UIElementType.RETURN_HOME_BUTTON.value: ["return_home_button.png", "return_home.png", "btn_return_home.png"],
                UIElementType.CLOSE_MODAL_BUTTON.value: ["close_modal_button.png", "close.png", "btn_close.png"],
            }

            for elem_type, filenames in template_mappings.items():
                for fname in filenames:
                    p = os.path.join(self.template_dir, fname)
                    if os.path.isfile(p):
                        img = cv2.imread(p)
                        if img is not None and img.size > 0:
                            self.register_template(elem_type, img, name=fname)

        # Ensure essential synthetic fallbacks are available if any are missing
        self._register_synthetic_fallbacks()

    def _register_synthetic_fallbacks(self) -> None:
        """Generates procedural reference templates if file assets are missing."""
        # 1. Next Button
        if UIElementType.NEXT_BUTTON.value not in self._templates:
            tpl = np.zeros((75, 180, 3), dtype=np.uint8)
            cv2.rectangle(tpl, (0, 0), (180, 75), (25, 140, 235), -1)
            cv2.rectangle(tpl, (0, 0), (180, 75), (255, 255, 255), 2)
            cv2.putText(tpl, "Next", (50, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            self.register_template(UIElementType.NEXT_BUTTON.value, tpl, name="synthetic_next")

        # 2. Battle Button
        if UIElementType.BATTLE_BUTTON.value not in self._templates:
            tpl = np.zeros((84, 87, 3), dtype=np.uint8)
            cv2.rectangle(tpl, (0, 0), (87, 84), (20, 180, 230), -1)
            cv2.rectangle(tpl, (0, 0), (87, 84), (255, 255, 255), 2)
            cv2.putText(tpl, "Battle", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            self.register_template(UIElementType.BATTLE_BUTTON.value, tpl, name="synthetic_battle")

        # 3. Find Match Button
        if UIElementType.FIND_MATCH_BUTTON.value not in self._templates:
            tpl = np.zeros((60, 193, 3), dtype=np.uint8)
            cv2.rectangle(tpl, (0, 0), (193, 60), (25, 160, 240), -1)
            cv2.rectangle(tpl, (0, 0), (193, 60), (255, 255, 255), 2)
            cv2.putText(tpl, "Find a Match", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            self.register_template(UIElementType.FIND_MATCH_BUTTON.value, tpl, name="synthetic_find_match")

        # 4. Attack Confirm Button (Green)
        if UIElementType.ATTACK_CONFIRM_BUTTON.value not in self._templates:
            tpl = np.zeros((37, 172, 3), dtype=np.uint8)
            cv2.rectangle(tpl, (0, 0), (172, 37), (46, 204, 113), -1)
            cv2.rectangle(tpl, (0, 0), (172, 37), (255, 255, 255), 2)
            cv2.putText(tpl, "Attack!", (35, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            self.register_template(UIElementType.ATTACK_CONFIRM_BUTTON.value, tpl, name="synthetic_attack")

        # 5. End Battle Button
        if UIElementType.END_BATTLE_BUTTON.value not in self._templates:
            tpl = np.zeros((70, 160, 3), dtype=np.uint8)
            cv2.rectangle(tpl, (0, 0), (160, 70), (40, 40, 200), -1)
            cv2.rectangle(tpl, (0, 0), (160, 70), (255, 255, 255), 2)
            cv2.putText(tpl, "End Battle", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            self.register_template(UIElementType.END_BATTLE_BUTTON.value, tpl, name="synthetic_end_battle")

        # 6. Return Home Button
        if UIElementType.RETURN_HOME_BUTTON.value not in self._templates:
            tpl = np.zeros((70, 180, 3), dtype=np.uint8)
            cv2.rectangle(tpl, (0, 0), (180, 70), (20, 170, 230), -1)
            cv2.rectangle(tpl, (0, 0), (180, 70), (255, 255, 255), 2)
            cv2.putText(tpl, "Return Home", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            self.register_template(UIElementType.RETURN_HOME_BUTTON.value, tpl, name="synthetic_return_home")

        # 7. Close Modal Button
        if UIElementType.CLOSE_MODAL_BUTTON.value not in self._templates:
            tpl = np.zeros((30, 30, 3), dtype=np.uint8)
            cv2.rectangle(tpl, (0, 0), (30, 30), (50, 50, 220), -1)
            cv2.line(tpl, (6, 6), (24, 24), (255, 255, 255), 2)
            cv2.line(tpl, (24, 6), (6, 24), (255, 255, 255), 2)
            self.register_template(UIElementType.CLOSE_MODAL_BUTTON.value, tpl, name="synthetic_close")

    def register_template(
        self,
        element_type: str,
        template: np.ndarray,
        name: str = "",
    ) -> None:
        """Registers a visual reference template for a specific UI element type."""
        if template is None or template.size == 0:
            return
        tpl_name = name or f"{element_type}_{len(self._templates.get(element_type, [])) + 1}"
        self._templates.setdefault(element_type, []).append((tpl_name, template))

    # --------------------------------------------------------------------------
    # Detection Pipeline
    # --------------------------------------------------------------------------

    def detect_elements(
        self,
        image: Optional[np.ndarray],
        current_state: Optional[GameState] = None,
        target_elements: Optional[List[str]] = None,
    ) -> List[UIElementResult]:
        """
        Executes multi-scale template matching across all registered UI elements.
        Detects ambiguous matches, eliminates spatial conflicts, and associates
        detections with game states.
        """
        t_start = time.perf_counter()

        if image is None or image.size == 0 or image.ndim < 2:
            self._last_diagnostics = DetectionDiagnostics(
                total_detections=0,
                execution_time_ms=0.0,
                frame_resolution=(0, 0),
                associated_state=current_state,
            )
            return []

        h, w = image.shape[:2]
        raw_candidates: List[UIElementResult] = []
        scores_by_element: Dict[str, float] = {}

        # Elements to scan for
        element_types = target_elements or list(self._templates.keys())

        # If current_state is provided, prioritize relevant element types
        if current_state is not None:
            relevant_types = [
                k.value for k, s in ELEMENT_STATE_MAP.items()
                if s == current_state
            ]
            # Put relevant types first for early evaluation
            element_types = sorted(
                element_types,
                key=lambda e: 0 if e in relevant_types else 1
            )

        for elem_type in element_types:
            templates = self._templates.get(elem_type, [])
            best_element_result: Optional[UIElementResult] = None
            elem_matches: List[Tuple[float, UIElementResult]] = []

            for tpl_name, tpl in templates:
                # Multi-scale match
                match_res = VisionPreprocessor.match_template_multiscale(
                    image=image,
                    template=tpl,
                    scales=self.config.scales,
                    method=self.config.method,
                    threshold=self.config.min_confidence,
                    early_exit_threshold=self.config.early_exit_threshold,
                )

                if match_res.found and match_res.confidence >= self.config.min_confidence:
                    # Optional color verification
                    if self.config.color_verification and not self._verify_color(image, match_res, elem_type):
                        continue

                    # Create normalized coordinates
                    bx, by, bw, bh = match_res.x, match_res.y, match_res.width, match_res.height
                    cx, cy = bx + bw // 2, by + bh // 2
                    norm_box = NormalizedRect(
                        x=bx / float(w),
                        y=by / float(h),
                        width=bw / float(w),
                        height=bh / float(h),
                    )
                    norm_center = NormalizedPoint(x=cx / float(w), y=cy / float(h))

                    # State association
                    state_assoc = None
                    for et, gs in ELEMENT_STATE_MAP.items():
                        if et.value == elem_type:
                            state_assoc = gs
                            break

                    res = UIElementResult(
                        element_type=elem_type,
                        bounding_box=Rect(bx, by, bw, bh),
                        normalized_box=norm_box,
                        center_point=Point(cx, cy),
                        normalized_center=norm_center,
                        confidence=match_res.confidence,
                        scale=match_res.scale,
                        state_association=state_assoc,
                        template_name=tpl_name,
                    )
                    elem_matches.append((match_res.confidence, res))

            if elem_matches:
                # Sort by confidence descending
                elem_matches.sort(key=lambda x: x[0], reverse=True)
                top_conf, top_res = elem_matches[0]
                scores_by_element[elem_type] = top_conf

                # Check for ambiguity if multiple templates or locations matched closely
                if len(elem_matches) > 1:
                    second_conf, second_res = elem_matches[1]
                    if (top_conf - second_conf) <= self.config.ambiguity_margin:
                        top_res.is_ambiguous = True
                        top_res.alternative_matches.append(second_res.to_dict())

                raw_candidates.append(top_res)

        # Resolve spatial conflicts (NMS / duplicate overlapping boxes)
        final_detections = self.resolve_conflicts(raw_candidates)

        # Telemetry diagnostics
        t_dur_ms = (time.perf_counter() - t_start) * 1000.0
        ambiguous_cnt = sum(1 for d in final_detections if d.is_ambiguous)
        conflict_cnt = len(raw_candidates) - len(final_detections)

        self._last_diagnostics = DetectionDiagnostics(
            total_detections=len(final_detections),
            execution_time_ms=t_dur_ms,
            elements_found=[d.element_type for d in final_detections],
            ambiguous_count=ambiguous_cnt,
            conflict_count=conflict_cnt,
            frame_resolution=(w, h),
            associated_state=current_state,
            scores_by_element=scores_by_element,
        )

        # Optional debug image persistence
        if self.config.save_debug_annotated_images and final_detections:
            self._save_auto_debug_image(image, final_detections)

        return final_detections

    def detect_element(
        self,
        image: Optional[np.ndarray],
        element_type: str,
        current_state: Optional[GameState] = None,
    ) -> Optional[UIElementResult]:
        """Detects a single specific UI element type."""
        results = self.detect_elements(
            image=image,
            current_state=current_state,
            target_elements=[element_type],
        )
        return results[0] if results else None

    def filter_by_state(
        self,
        results: List[UIElementResult],
        state: GameState,
    ) -> List[UIElementResult]:
        """Filters detections to those associated with the given GameState."""
        return [r for r in results if r.state_association == state]

    def resolve_conflicts(
        self,
        candidates: List[UIElementResult],
    ) -> List[UIElementResult]:
        """
        Resolves spatial overlaps and conflicts via Non-Maximum Suppression (NMS).
        When two detections overlap significantly (IoU > threshold), the one with
        higher confidence score is retained.
        """
        if not candidates or len(candidates) <= 1:
            return list(candidates)

        # Sort by confidence descending
        sorted_cand = sorted(candidates, key=lambda c: c.confidence, reverse=True)
        kept: List[UIElementResult] = []

        for cand in sorted_cand:
            b1 = cand.bounding_box
            suppressed = False

            for accepted in kept:
                b2 = accepted.bounding_box
                iou = self._compute_iou(b1, b2)
                if iou >= self.config.nms_iou_threshold:
                    # Conflict detected: keep the higher confidence one
                    suppressed = True
                    break

            if not suppressed:
                kept.append(cand)

        return kept

    @staticmethod
    def _compute_iou(box1: Rect, box2: Rect) -> float:
        """Computes Intersection over Union (IoU) of two bounding boxes."""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)

        intersection_w = max(0, x2 - x1)
        intersection_h = max(0, y2 - y1)
        intersection_area = intersection_w * intersection_h

        if intersection_area <= 0:
            return 0.0

        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union_area = float(area1 + area2 - intersection_area)

        return (intersection_area / union_area) if union_area > 0 else 0.0

    def _verify_color(
        self,
        image: np.ndarray,
        match: TemplateMatchResult,
        element_type: str,
    ) -> bool:
        """
        Validates dominant color profile of the detected crop in HSV space.
        Prevents false positive correlations on similarly-shaped gray/dark UI elements.
        """
        try:
            bx, by, bw, bh = match.x, match.y, match.width, match.height
            h, w = image.shape[:2]
            crop = image[max(0, by):min(h, by + bh), max(0, bx):min(w, bx + bw)]
            if crop.size == 0:
                return True

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            # Attack! button must contain vivid green pixels
            if element_type == UIElementType.ATTACK_CONFIRM_BUTTON.value:
                # Green in HSV: H: 35..85, S: 70..255, V: 70..255
                green_mask = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([85, 255, 255]))
                return bool(cv2.countNonZero(green_mask) >= 15)

            # Next button must contain orange/yellow pixels
            if element_type == UIElementType.NEXT_BUTTON.value:
                # Orange/Yellow in HSV: H: 10..30, S: 100..255, V: 100..255
                orange_mask = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([30, 255, 255]))
                return bool(cv2.countNonZero(orange_mask) >= 20)

            # End Battle / Close button must contain red pixels
            if element_type in (UIElementType.END_BATTLE_BUTTON.value, UIElementType.CLOSE_MODAL_BUTTON.value):
                # Red wraps around 0 and 180 in HSV
                r1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
                r2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
                red_mask = cv2.bitwise_or(r1, r2)
                return bool(cv2.countNonZero(red_mask) >= 10)

            # Battle / Find Match button must contain warm gold/orange pixels
            if element_type in (UIElementType.BATTLE_BUTTON.value, UIElementType.FIND_MATCH_BUTTON.value):
                gold_mask = cv2.inRange(hsv, np.array([12, 80, 80]), np.array([35, 255, 255]))
                return bool(cv2.countNonZero(gold_mask) >= 20)

        except Exception as ex:
            logger.debug("Color verification error: %s", ex)
            return True

        return True

    # --------------------------------------------------------------------------
    # Overlay & Visualization
    # --------------------------------------------------------------------------

    def annotate_frame(
        self,
        image: np.ndarray,
        detections: List[UIElementResult],
        diagnostics: Optional[DetectionDiagnostics] = None,
        show_hud: bool = True,
    ) -> np.ndarray:
        """
        Renders bounding boxes, confidence tags, and diagnostics on a copy of the frame.
        Guarantees zero mutation of the source image.
        """
        if image is None or image.size == 0:
            return image

        annotated = image.copy()
        h, w = annotated.shape[:2]

        for det in detections:
            box = det.bounding_box
            color = ELEMENT_COLOR_MAP.get(det.element_type, (0, 255, 0))

            # Bounding rectangle
            p1 = (box.x, box.y)
            p2 = (box.x + box.width, box.y + box.height)
            thickness = 2
            cv2.rectangle(annotated, p1, p2, color, thickness)

            # Center target crosshair
            cx, cy = det.center_point.x, det.center_point.y
            cv2.drawMarker(annotated, (cx, cy), color, cv2.MARKER_CROSS, 8, 1)

            # Label text
            status_tag = " [AMBIGUOUS]" if det.is_ambiguous else ""
            label = f"{det.element_type} ({det.confidence * 100:.1f}%){status_tag}"

            # Label background pill
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
            label_y = max(th + 4, box.y - 4)
            cv2.rectangle(
                annotated,
                (box.x, label_y - th - 3),
                (box.x + tw + 6, label_y + baseline),
                (20, 24, 30),
                -1
            )
            cv2.putText(
                annotated,
                label,
                (box.x + 3, label_y - 2),
                font,
                font_scale,
                color,
                1,
                cv2.LINE_AA
            )

        # Diagnostic HUD overlay on top-left
        if show_hud:
            diag = diagnostics or self._last_diagnostics
            if diag is not None:
                hud_text = f"UI Detections: {len(detections)} | Time: {diag.execution_time_ms:.1f}ms"
                if diag.associated_state:
                    hud_text += f" | State: {diag.associated_state.value}"
                cv2.rectangle(annotated, (10, 10), (320, 36), (15, 18, 25), -1)
                cv2.rectangle(annotated, (10, 10), (320, 36), (0, 210, 211), 1)
                cv2.putText(
                    annotated,
                    hud_text,
                    (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (220, 230, 240),
                    1,
                    cv2.LINE_AA
                )

        return annotated

    def save_annotated_frame(
        self,
        image: np.ndarray,
        detections: List[UIElementResult],
        output_path: str,
        diagnostics: Optional[DetectionDiagnostics] = None,
    ) -> str:
        """Saves an annotated visual debug frame to disk."""
        annotated = self.annotate_frame(image, detections, diagnostics)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, annotated)
        return output_path

    def _save_auto_debug_image(
        self,
        image: np.ndarray,
        detections: List[UIElementResult],
    ) -> None:
        """Saves automated debug frame during live inspection."""
        try:
            os.makedirs(self.config.debug_output_dir, exist_ok=True)
            fname = f"detection_{int(time.time() * 1000)}.jpg"
            out_p = os.path.join(self.config.debug_output_dir, fname)
            self.save_annotated_frame(image, detections, out_p)
        except Exception as ex:
            logger.debug("Failed saving debug detection image: %s", ex)

    def get_diagnostics(self) -> Optional[DetectionDiagnostics]:
        """Returns the telemetry diagnostics from the most recent detection run."""
        return self._last_diagnostics
