"""
ClashVision Vision Subsystem — Base Layout Analyzer.
Extracts tactical structural features from enemy base layouts:
- Exposed collector ratio (exposed collectors vs core storages)
- Town Hall accessibility (snipable / perimeter vs core fortified)
- Air Defense spatial coordinates
- Wall compartment estimation
Emits typed BaseLayoutFeatures for downstream farming decision engines.
Derived from Requirement R3 and Feature 15.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from clashvision.core.models import BaseLayoutFeatures


class BaseLayoutAnalyzer:
    """
    Extracts tactical spatial layout features from either:
    1. A BaseLayoutModel instance (from offline fixtures), or
    2. A raw BGR image / top-down tactical base map.
    """

    def __init__(self, reference_width: int = 1920, reference_height: int = 1080) -> None:
        self.ref_w = reference_width
        self.ref_h = reference_height

    def analyze_layout(self, source: Any) -> BaseLayoutFeatures:
        """
        Universal entry point: analyzes a BaseLayoutModel or image frame.
        """
        if source is None:
            return BaseLayoutFeatures()

        # Check if source is a BaseLayoutModel or object with building entities (duck typing)
        if hasattr(source, "buildings") or hasattr(source, "wall_coordinates"):
            return self._analyze_model(source)

        if isinstance(source, np.ndarray):
            return self._analyze_image(source)

        return BaseLayoutFeatures()

    def _analyze_model(self, model: Any) -> BaseLayoutFeatures:
        """
        Extracts features directly from a BaseLayoutModel instance.
        """
        total_collectors = 0
        exposed_collectors = 0
        core_resources = 0
        perimeter_resources = 0
        air_defenses: List[Tuple[int, int]] = []
        th_accessible = getattr(model, "th_accessible", False)
        th_level: Optional[int] = None
        has_walls = len(getattr(model, "wall_coordinates", [])) > 0
        center_x = self.ref_w / 2.0
        center_y = self.ref_h / 2.0
        max_dist = math.sqrt(center_x ** 2 + center_y ** 2)

        for b in getattr(model, "buildings", []):
            try:
                bx = float(b.x)
                by = float(b.y)
                if not (math.isfinite(bx) and math.isfinite(by)):
                    continue
            except (TypeError, ValueError, AttributeError):
                continue

            # Convert normalized [0, 1] to reference pixels if necessary
            px = int(round(bx * self.ref_w)) if 0.0 <= bx <= 1.0 else int(round(bx))
            py = int(round(by * self.ref_h)) if 0.0 <= by <= 1.0 else int(round(by))
            b_name = getattr(b, "name", "").lower()

            is_coll = getattr(b, "is_collector", False) or any(k in b_name for k in ("collector", "mine", "drill"))
            is_stor = getattr(b, "is_storage", False) or "storage" in b_name

            dist = math.sqrt((px - center_x) ** 2 + (py - center_y) ** 2)
            is_perim = dist > (max_dist * 0.32) or getattr(b, "is_exposed", False)

            if is_coll:
                total_collectors += 1
                if hasattr(b, "is_exposed"):
                    if getattr(b, "is_exposed", False):
                        exposed_collectors += 1
                elif is_perim:
                    exposed_collectors += 1

            if is_coll or is_stor:
                if is_perim:
                    perimeter_resources += 1
                else:
                    core_resources += 1

            if getattr(b, "is_air_defense", False):
                air_defenses.append((px, py))

            if getattr(b, "is_town_hall", False):
                th_level = getattr(b, "hit_points", 1000) // 1000 + 10  # approximate level

        if total_collectors > 0:
            ratio = float(exposed_collectors) / float(total_collectors)
        else:
            ratio = float(getattr(model, "exposed_collector_ratio", 0.0))

        # If model explicitly provides air defense locations
        if not air_defenses and hasattr(model, "air_defense_locations"):
            for ad_x, ad_y in model.air_defense_locations:
                try:
                    fad_x, fad_y = float(ad_x), float(ad_y)
                    if not (math.isfinite(fad_x) and math.isfinite(fad_y)):
                        continue
                    px = int(round(fad_x * self.ref_w)) if 0.0 <= fad_x <= 1.0 else int(round(fad_x))
                    py = int(round(fad_y * self.ref_h)) if 0.0 <= fad_y <= 1.0 else int(round(fad_y))
                    air_defenses.append((px, py))
                except (TypeError, ValueError):
                    continue

        # Estimate compartments from wall count
        wall_coords = getattr(model, "wall_coordinates", [])
        compartment_count = max(1, len(wall_coords) // 25) if has_walls else 0

        total_res = core_resources + perimeter_resources
        if total_res > 0:
            core_ratio = float(core_resources) / float(total_res)
            perim_ratio = float(perimeter_resources) / float(total_res)
        else:
            if getattr(model, "dead_base", False):
                core_ratio, perim_ratio = 0.15, 0.85
            else:
                core_ratio = 0.50
                perim_ratio = ratio

        access_score = max(0.0, min(1.0, 0.85 * perim_ratio + 0.15 * max(0.0, 1.0 - (compartment_count / 10.0))))
        symm_score = self.compute_base_symmetry(model)

        # Tactical entry points & defense hazards
        entry_sectors = self.evaluate_entry_points(model)
        defense_hazards = {s["sector"]: s["hazard_score"] for s in entry_sectors}

        # Base difficulty score [0.0, 1.0]
        comp_factor = min(1.0, compartment_count / 8.0)
        col_factor = 1.0 - ratio
        th_factor = 0.2 if th_accessible else 0.8
        diff_score = 0.35 * comp_factor + 0.35 * col_factor + 0.20 * th_factor + 0.10 * (1.0 - access_score)
        diff_score = float(max(0.05, min(0.98, diff_score)))

        return BaseLayoutFeatures(
            exposed_collector_ratio=round(ratio, 3),
            th_accessible=bool(th_accessible),
            th_level=th_level or 14,
            air_defense_locations=air_defenses,
            compartment_count=compartment_count,
            wall_perimeter_intact=has_walls,
            core_resource_ratio=round(core_ratio, 3),
            perimeter_resource_ratio=round(perim_ratio, 3),
            resource_accessibility_score=round(access_score, 3),
            symmetry_score=round(symm_score, 3),
            difficulty_score=round(diff_score, 3),
            entry_points=entry_sectors,
            defense_hazards=defense_hazards,
            metadata={
                "layout_type": str(getattr(model, "layout_type", "UNKNOWN")),
                "total_collectors": total_collectors,
                "exposed_collectors": exposed_collectors,
                "core_resources": core_resources,
                "perimeter_resources": perimeter_resources,
                "total_walls": len(wall_coords),
                "storages_in_core": getattr(model, "storages_in_core", True),
                "dead_base": getattr(model, "dead_base", False),
            },
        )

    def _analyze_image(self, image: np.ndarray) -> BaseLayoutFeatures:
        """
        Computer vision feature extraction on a 2D base map or rendered tactical canvas.
        """
        if image is None or image.size == 0 or len(image.shape) < 2:
            return BaseLayoutFeatures()

        # Convert non-uint8 dtypes safely to uint8 for OpenCV compatibility
        if image.dtype != np.uint8:
            if image.dtype == bool:
                image = (image.astype(np.uint8) * 255)
            elif np.issubdtype(image.dtype, np.floating):
                clean_img = np.nan_to_num(image, copy=True, nan=0.0, posinf=255.0, neginf=0.0)
                if clean_img.size > 0 and clean_img.max() <= 1.0 and clean_img.min() >= 0.0 and clean_img.max() > 0.0:
                    image = np.clip(clean_img * 255.0, 0, 255).astype(np.uint8)
                else:
                    image = np.clip(clean_img, 0, 255).astype(np.uint8)
            elif image.dtype == np.uint16:
                if image.size > 0 and image.max() > 255:
                    image = (image / 256).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        max_dist_from_center = math.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2)

        # Standardize 3-channel BGR
        if len(image.shape) == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] == 1:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] > 4:
            bgr = image[:, :, :3]
        else:
            bgr = image

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # 1. Detect Air Defenses (Cyan/metallic color: H: 80-110, S: 100-255, V: 100-255)
        ad_mask = cv2.inRange(hsv, np.array([80, 100, 100]), np.array([110, 255, 255]))
        ad_contours, _ = cv2.findContours(ad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        air_defenses: List[Tuple[int, int]] = []
        for c in ad_contours:
            if cv2.contourArea(c) > 20:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    air_defenses.append((cx, cy))

        # 2. Detect Town Hall (Red/Dark building: H: 0-15 or 165-180, S: 120-255, V: 100-255)
        th_mask1 = cv2.inRange(hsv, np.array([0, 120, 100]), np.array([10, 255, 255]))
        th_mask2 = cv2.inRange(hsv, np.array([165, 120, 100]), np.array([180, 255, 255]))
        th_mask = cv2.bitwise_or(th_mask1, th_mask2)
        th_contours, _ = cv2.findContours(th_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        th_accessible = False
        th_pos: Optional[Tuple[int, int]] = None
        valid_th_contours = [c for c in th_contours if cv2.contourArea(c) > 100]
        if valid_th_contours:
            best_th = max(valid_th_contours, key=cv2.contourArea)
            M = cv2.moments(best_th)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                th_pos = (cx, cy)
                dist_to_center = math.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
                # If TH is situated in outer 35% of the base area -> accessible
                if dist_to_center > (max_dist_from_center * 0.35):
                    th_accessible = True

        # 3. Detect Collectors (Yellow Gold and Magenta Elixir)
        # Gold: H: 20-35, S: 120-255, V: 120-255
        gold_mask = cv2.inRange(hsv, np.array([20, 120, 120]), np.array([35, 255, 255]))
        # Magenta: H: 140-165, S: 100-255, V: 100-255
        elixir_mask = cv2.inRange(hsv, np.array([140, 100, 100]), np.array([165, 255, 255]))
        collector_mask = cv2.bitwise_or(gold_mask, elixir_mask)

        col_contours, _ = cv2.findContours(collector_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_collectors = 0
        exposed_collectors = 0

        for c in col_contours:
            if cv2.contourArea(c) > 30:
                total_collectors += 1
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dist = math.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
                    # Outer perimeter collectors
                    if dist > (max_dist_from_center * 0.30):
                        exposed_collectors += 1

        exp_ratio = float(exposed_collectors) / float(total_collectors) if total_collectors > 0 else 0.0
        core_collectors = total_collectors - exposed_collectors
        core_ratio = float(core_collectors) / float(total_collectors) if total_collectors > 0 else 0.5
        perim_ratio = exp_ratio

        # 4. Detect Walls (Gray pixels: S < 40, 120 < V < 220)
        wall_mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 40, 220]))
        wall_contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        compartments = sum(1 for c in wall_contours if cv2.contourArea(c) > 150)

        access_score = max(0.0, min(1.0, 0.85 * perim_ratio + 0.15 * max(0.0, 1.0 - (compartments / 10.0))))
        symm_score = self.compute_base_symmetry(image)

        # Tactical entry points & defense hazards for image
        entry_sectors = self.evaluate_entry_points(image)
        defense_hazards = {s["sector"]: s["hazard_score"] for s in entry_sectors}

        comp_factor = min(1.0, compartments / 8.0)
        col_factor = 1.0 - exp_ratio
        th_factor = 0.2 if th_accessible else 0.8
        diff_score = 0.35 * comp_factor + 0.35 * col_factor + 0.20 * th_factor + 0.10 * (1.0 - access_score)
        diff_score = float(max(0.05, min(0.98, diff_score)))

        return BaseLayoutFeatures(
            exposed_collector_ratio=round(exp_ratio, 3),
            th_accessible=th_accessible,
            th_level=14,
            air_defense_locations=air_defenses,
            compartment_count=max(1, compartments),
            wall_perimeter_intact=compartments > 0,
            core_resource_ratio=round(core_ratio, 3),
            perimeter_resource_ratio=round(perim_ratio, 3),
            resource_accessibility_score=round(access_score, 3),
            symmetry_score=round(symm_score, 3),
            difficulty_score=round(diff_score, 3),
            entry_points=entry_sectors,
            defense_hazards=defense_hazards,
            metadata={
                "detected_collectors": total_collectors,
                "detected_exposed": exposed_collectors,
                "detected_core": core_collectors,
                "th_coordinates": th_pos,
                "air_defense_count": len(air_defenses),
            },
        )

    def compute_base_symmetry(self, source: Any) -> float:
        """
        Quantifies structural base symmetry in range [0.0, 1.0].
        Evaluates horizontal reflection, vertical reflection, and 180-deg rotational symmetry.
        """
        if source is None:
            return 0.0

        # Case A: BaseLayoutModel or object with buildings/coordinates
        if hasattr(source, "buildings") or hasattr(source, "wall_coordinates"):
            points: List[Tuple[float, float]] = []
            for b in getattr(source, "buildings", []):
                try:
                    bx, by = float(b.x), float(b.y)
                    if math.isfinite(bx) and math.isfinite(by):
                        points.append((bx, by))
                except (TypeError, ValueError, AttributeError):
                    pass

            for w in getattr(source, "wall_coordinates", []):
                try:
                    wx, wy = float(w[0]), float(w[1])
                    if math.isfinite(wx) and math.isfinite(wy):
                        points.append((wx, wy))
                except (TypeError, ValueError, IndexError):
                    pass

            if len(points) < 4:
                return 0.5  # Neutral default for sparse layouts

            # Determine centroid
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            tolerance = max(0.04 * max(span_x, span_y), 0.03)

            # Check matching for horizontal (flip x across cx), vertical (flip y across cy), rotational
            pts_set = points
            h_matches = 0
            v_matches = 0
            r_matches = 0

            for px, py in pts_set:
                target_hx, target_hy = 2 * cx - px, py
                target_vx, target_vy = px, 2 * cy - py
                target_rx, target_ry = 2 * cx - px, 2 * cy - py

                # Nearest neighbor check
                if any(abs(ox - target_hx) <= tolerance and abs(oy - target_hy) <= tolerance for ox, oy in pts_set):
                    h_matches += 1
                if any(abs(ox - target_vx) <= tolerance and abs(oy - target_vy) <= tolerance for ox, oy in pts_set):
                    v_matches += 1
                if any(abs(ox - target_rx) <= tolerance and abs(oy - target_ry) <= tolerance for ox, oy in pts_set):
                    r_matches += 1

            n = float(len(pts_set))
            score_h = h_matches / n
            score_v = v_matches / n
            score_r = r_matches / n
            return float(max(0.0, min(1.0, max(score_h, score_v, score_r))))

        # Case B: Image ndarray
        if isinstance(source, np.ndarray) and source.size > 0 and len(source.shape) >= 2:
            gray = source if len(source.shape) == 2 else cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            # Resize to standard analysis size for speed and normalized comparison
            resized = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)

            flip_h = cv2.flip(resized, 1)
            flip_v = cv2.flip(resized, 0)
            flip_r = cv2.flip(resized, -1)

            corr_h = cv2.matchTemplate(resized, flip_h, cv2.TM_CCOEFF_NORMED)[0][0]
            corr_v = cv2.matchTemplate(resized, flip_v, cv2.TM_CCOEFF_NORMED)[0][0]
            corr_r = cv2.matchTemplate(resized, flip_r, cv2.TM_CCOEFF_NORMED)[0][0]

            best_corr = max(0.0, float(max(corr_h, corr_v, corr_r)))
            return float(min(1.0, best_corr))

        return 0.0

    def analyze_resource_distribution(self, source: Any) -> Dict[str, float]:
        """
        Direct convenience method for evaluating resource accessibility and distribution.
        """
        features = self.analyze_layout(source)
        return {
            "core_resource_ratio": features.core_resource_ratio,
            "perimeter_resource_ratio": features.perimeter_resource_ratio,
            "resource_accessibility_score": features.resource_accessibility_score,
            "exposed_collector_ratio": features.exposed_collector_ratio,
        }

    def evaluate_entry_points(self, source: Any) -> List[Dict[str, Any]]:
        """
        Evaluates 4 primary directional entry sectors (North, East, South, West).
        Estimates defense hazard, resource proximity, and flags the optimal entry point.
        """
        sectors = [
            {"sector": "North", "coord": (int(self.ref_w * 0.5), int(self.ref_h * 0.12)), "hazard": 0.2, "defenses": 0, "resources": 0},
            {"sector": "East", "coord": (int(self.ref_w * 0.88), int(self.ref_h * 0.5)), "hazard": 0.2, "defenses": 0, "resources": 0},
            {"sector": "South", "coord": (int(self.ref_w * 0.5), int(self.ref_h * 0.88)), "hazard": 0.2, "defenses": 0, "resources": 0},
            {"sector": "West", "coord": (int(self.ref_w * 0.12), int(self.ref_h * 0.5)), "hazard": 0.2, "defenses": 0, "resources": 0},
        ]
        if source is None:
            return [
                {
                    "sector": s["sector"],
                    "hazard_score": 0.2,
                    "entry_coordinate": s["coord"],
                    "defenses_count": 0,
                    "resources_count": 0,
                    "recommended": s["sector"] == "South",
                }
                for s in sectors
            ]

        # Model source
        if hasattr(source, "buildings"):
            cx = 0.5
            cy = 0.5
            for b in getattr(source, "buildings", []):
                try:
                    bx = float(b.x) if float(b.x) <= 1.0 else float(b.x) / self.ref_w
                    by = float(b.y) if float(b.y) <= 1.0 else float(b.y) / self.ref_h
                except (TypeError, ValueError, AttributeError):
                    continue

                dx = bx - cx
                dy = by - cy
                # Determine sector by dominant offset
                if abs(dx) > abs(dy):
                    target_sec = "East" if dx > 0 else "West"
                else:
                    target_sec = "South" if dy > 0 else "North"

                sec_entry = next(s for s in sectors if s["sector"] == target_sec)
                b_name = getattr(b, "name", "").lower()
                is_def = getattr(b, "is_defense", False) or getattr(b, "is_air_defense", False) or any(d in b_name for d in ("cannon", "tower", "mortar", "defense", "inferno", "eagle", "xbow"))
                is_res = getattr(b, "is_collector", False) or getattr(b, "is_storage", False) or any(r in b_name for r in ("mine", "collector", "drill", "storage"))
                is_high_hazard = getattr(b, "is_air_defense", False) or any(h in b_name for h in ("inferno", "eagle", "scattershot"))

                if is_def:
                    sec_entry["defenses"] += 1
                    sec_entry["hazard"] += 0.25 if is_high_hazard else 0.10
                if is_res:
                    sec_entry["resources"] += 1

        elif isinstance(source, np.ndarray) and source.size > 0:
            # Uniform baseline hazard for raw image frames
            pass

        # Normalize hazards and pick recommended sector
        results = []
        best_score = -999.0
        best_idx = 0
        for i, s in enumerate(sectors):
            norm_hazard = float(max(0.05, min(0.95, round(s["hazard"], 2))))
            suitability = (s["resources"] * 1.5) - (norm_hazard * 3.0)
            if suitability > best_score:
                best_score = suitability
                best_idx = i

        for i, s in enumerate(sectors):
            results.append({
                "sector": s["sector"],
                "hazard_score": float(max(0.05, min(0.95, round(s["hazard"], 2)))),
                "entry_coordinate": s["coord"],
                "defenses_count": s["defenses"],
                "resources_count": s["resources"],
                "recommended": (i == best_idx),
            })
        return results

    def compute_difficulty_score(self, source: Any, features: Optional[BaseLayoutFeatures] = None) -> float:
        """
        Calculates an aggregate base difficulty rating [0.0, 1.0].
        """
        if features is None:
            features = self.analyze_layout(source)

        comp_factor = min(1.0, features.compartment_count / 8.0)
        col_factor = 1.0 - features.exposed_collector_ratio
        th_factor = 0.2 if features.th_accessible else 0.8
        diff_score = 0.35 * comp_factor + 0.35 * col_factor + 0.20 * th_factor + 0.10 * (1.0 - features.resource_accessibility_score)
        return float(max(0.05, min(0.98, round(diff_score, 3))))
