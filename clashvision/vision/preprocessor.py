"""
ClashVision Vision Subsystem — Image Preprocessor.
Provides robust grayscale conversion, ROI cropping, adaptive thresholding,
contrast enhancement (CLAHE), morphological noise filtering, and visual OCR debug models.
Derived from Requirement R2 and Features 9, 10, 13.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import time
from typing import Optional, Tuple, Union, Dict, Any

import cv2
import numpy as np

from clashvision.core.models import Rect, NormalizedRect


@dataclass
class OCRDebugData:
    """
    Visual OCR debug model encapsulating raw crop, preprocessed thresholded image,
    raw OCR recognition string, and parsed integer (Requirement R2 / Feature 13).
    """
    raw_crop: np.ndarray
    preprocessed_crop: np.ndarray
    raw_text: str = ""
    parsed_value: Optional[int] = None
    confidence: float = 0.0
    roi_name: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roi_name": self.roi_name,
            "raw_text": self.raw_text,
            "parsed_value": self.parsed_value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "raw_crop_shape": list(self.raw_crop.shape) if self.raw_crop is not None else [],
            "preprocessed_crop_shape": list(self.preprocessed_crop.shape) if self.preprocessed_crop is not None else [],
        }


@dataclass
class TemplateMatchResult:
    """
    Structured outcome of a template matching operation.
    """
    found: bool
    confidence: float
    x: int
    y: int
    width: int
    height: int
    scale: float
    method: str = "grayscale"

    @property
    def rect(self) -> Rect:
        """Returns bounding rectangle in pixel space."""
        return Rect(x=self.x, y=self.y, width=self.width, height=self.height)

    @property
    def center(self) -> Tuple[int, int]:
        """Returns integer center coordinate (cx, cy)."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "confidence": round(float(self.confidence), 4),
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
            "scale": round(float(self.scale), 3),
            "method": self.method,
        }


class VisionPreprocessor:
    """
    High-performance image preprocessor tailored for Clash of Clans UI recognition
    and numeric resource OCR.
    """

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Converts any image (BGR, BGRA, or single-channel) to uint8 grayscale.
        Guarantees 2D uint8 output.
        """
        if image is None or image.size == 0 or image.ndim < 2:
            return np.zeros((0, 0), dtype=np.uint8)

        # Convert non-uint8 dtypes to uint8
        if image.dtype != np.uint8:
            if image.dtype == bool:
                img_u8 = (image.astype(np.uint8) * 255)
            elif np.issubdtype(image.dtype, np.floating):
                # Float images: sanitize NaNs/Infs, if normalized in [0, 1] with positive values, scale by 255
                clean_img = np.nan_to_num(image, copy=True, nan=0.0, posinf=255.0, neginf=0.0)
                if clean_img.size > 0 and clean_img.max() <= 1.0 and clean_img.min() >= 0.0 and clean_img.max() > 0.0:
                    img_u8 = np.clip(clean_img * 255.0, 0, 255).astype(np.uint8)
                else:
                    img_u8 = np.clip(clean_img, 0, 255).astype(np.uint8)
            elif image.dtype == np.uint16:
                if image.size > 0 and image.max() > 255:
                    img_u8 = (image / 256).astype(np.uint8)
                else:
                    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
            else:
                img_u8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_u8 = image

        if len(img_u8.shape) == 2:
            return img_u8

        channels = img_u8.shape[2]
        if channels == 3:
            return cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            return cv2.cvtColor(img_u8, cv2.COLOR_BGRA2GRAY)
        elif channels == 1:
            return img_u8.reshape(img_u8.shape[0], img_u8.shape[1])
        elif channels > 4:
            return cv2.cvtColor(img_u8[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            return img_u8[:, :, 0].copy()

    @staticmethod
    def crop_roi(
        image: np.ndarray,
        roi: Union[Rect, NormalizedRect, Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """
        Safely extracts an ROI crop from an image, clamping strictly to image boundaries.
        Returns empty array if image is None or dimensions are invalid.
        """
        if image is None or image.size == 0 or image.ndim < 2:
            empty_shape = (0, 0, image.shape[2]) if image is not None and image.ndim == 3 else (0, 0)
            return np.zeros(empty_shape, dtype=np.uint8 if image is None else image.dtype)

        img_h, img_w = image.shape[:2]

        if isinstance(roi, NormalizedRect):
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (roi.x, roi.y, roi.width, roi.height)):
                empty_shape = (0, 0, image.shape[2]) if image.ndim == 3 else (0, 0)
                return np.zeros(empty_shape, dtype=image.dtype)
            pixel_rect = roi.to_pixel_rect(img_w, img_h)
            x, y, w, h = pixel_rect.x, pixel_rect.y, pixel_rect.width, pixel_rect.height
        elif isinstance(roi, Rect):
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (roi.x, roi.y, roi.width, roi.height)):
                empty_shape = (0, 0, image.shape[2]) if image.ndim == 3 else (0, 0)
                return np.zeros(empty_shape, dtype=image.dtype)
            x, y, w, h = roi.x, roi.y, roi.width, roi.height
        else:
            try:
                x, y, w, h = roi
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y, w, h)):
                    empty_shape = (0, 0, image.shape[2]) if image.ndim == 3 else (0, 0)
                    return np.zeros(empty_shape, dtype=image.dtype)
            except Exception:
                empty_shape = (0, 0, image.shape[2]) if image.ndim == 3 else (0, 0)
                return np.zeros(empty_shape, dtype=image.dtype)

        # Clamp bounds
        x1 = max(0, min(img_w, int(x)))
        y1 = max(0, min(img_h, int(y)))
        x2 = max(x1, min(img_w, int(x + w)))
        y2 = max(y1, min(img_h, int(y + h)))

        if x2 <= x1 or y2 <= y1:
            # Fallback zero-dimension slice
            if len(image.shape) == 3:
                return np.zeros((0, 0, image.shape[2]), dtype=image.dtype)
            return np.zeros((0, 0), dtype=image.dtype)

        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def enhance_contrast(
        gray: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        Enhances local contrast using Contrast Limited Adaptive Histogram Equalization (CLAHE).
        """
        if gray is None or gray.size == 0:
            return gray
        if len(gray.shape) > 2 or gray.dtype != np.uint8:
            gray = VisionPreprocessor.to_grayscale(gray)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray)

    @staticmethod
    def adaptive_threshold(
        gray: np.ndarray,
        method: str = "otsu",
        block_size: int = 11,
        c: int = 2,
        invert: bool = False,
    ) -> np.ndarray:
        """
        Applies binarization thresholding:
        - 'otsu': Otsu's optimal global clustering threshold
        - 'gaussian': Local adaptive Gaussian thresholding
        - 'mean': Local adaptive mean thresholding
        """
        if gray is None or gray.size == 0:
            return gray
        if len(gray.shape) > 2 or gray.dtype != np.uint8:
            gray = VisionPreprocessor.to_grayscale(gray)

        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

        method_lower = method.lower()
        if method_lower == "otsu":
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)
            return thresh
        elif method_lower == "gaussian":
            # Ensure odd block size >= 3
            bs = max(3, block_size if block_size % 2 == 1 else block_size + 1)
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, bs, c)
        elif method_lower == "mean":
            bs = max(3, block_size if block_size % 2 == 1 else block_size + 1)
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, thresh_type, bs, c)
        else:
            # Fixed 128 threshold
            _, thresh = cv2.threshold(gray, 128, 255, thresh_type)
            return thresh

    @staticmethod
    def remove_noise(
        binary: np.ndarray,
        kernel_size: int = 2,
        operation: str = "open",
    ) -> np.ndarray:
        """
        Applies morphological operations (opening, closing, dilate, erode) for noise cleaning.
        """
        if binary is None or binary.size == 0:
            return binary

        k_size = max(1, kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

        op_lower = operation.lower()
        if op_lower == "open":
            return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif op_lower == "close":
            return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif op_lower == "dilate":
            return cv2.dilate(binary, kernel, iterations=1)
        elif op_lower == "erode":
            return cv2.erode(binary, kernel, iterations=1)
        return binary

    @staticmethod
    def scale_for_ocr(
        image: np.ndarray,
        scale_factor: float = 2.0,
        interpolation: int = cv2.INTER_CUBIC,
    ) -> np.ndarray:
        """
        Upscales small crops to boost OCR character recognition rates.
        """
        if image is None or image.size == 0 or scale_factor == 1.0:
            return image
        new_w = max(1, int(round(image.shape[1] * scale_factor)))
        new_h = max(1, int(round(image.shape[0] * scale_factor)))
        return cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    @classmethod
    def preprocess_for_ocr(
        cls,
        crop: np.ndarray,
        enhance: bool = True,
        binarize: bool = True,
        invert: bool = False,
        scale: float = 1.0,
        noise_removal: bool = True,
    ) -> np.ndarray:
        """
        End-to-end OCR preprocessing pipeline:
        1. Grayscale conversion
        2. Contrast enhancement via CLAHE
        3. Gaussian smoothing & Otsu adaptive binarization
        4. Morphological noise opening
        5. Optional inversion and scaling
        """
        if crop is None or crop.size == 0:
            return crop

        # 1. Grayscale
        gray = cls.to_grayscale(crop)

        # 2. Contrast enhancement
        if enhance:
            gray = cls.enhance_contrast(gray)

        if not binarize:
            if scale > 1.0:
                return cls.scale_for_ocr(gray, scale)
            return gray

        # 3. Thresholding
        thresh = cls.adaptive_threshold(gray, method="otsu", invert=invert)

        # 4. Noise removal
        if noise_removal:
            thresh = cls.remove_noise(thresh, kernel_size=2, operation="open")

        # 5. Scaling
        if scale > 1.0:
            thresh = cls.scale_for_ocr(thresh, scale)

        return thresh

    @classmethod
    def preprocess_resource_crop(
        cls,
        crop: np.ndarray,
        resource_type: str = "gold",
        strategy: str = "auto",
        method: Optional[str] = None,
        target_height: int = 40,
    ) -> np.ndarray:
        """
        Specialized multi-strategy preprocessing tailored for in-game resource meters.
        Supports:
        - 'clahe_otsu': Contrast-limited adaptive equalization with Otsu binarization.
        - 'adaptive_gaussian': Local adaptive Gaussian thresholding for dark or shaded backgrounds.
        - 'luminance_mask': High-value brightness isolation (isolates white digits from drop-shadows).
        - 'bilateral_adaptive': Noise reduction + adaptive threshold.
        - 'invert_thresh': Inverted thresholding.
        - 'auto': Resource-specific heuristic selection based on resource_type.
        """
        if crop is None or crop.size == 0:
            return crop

        effective_strat = (method or strategy or "auto").lower()

        # Upscale if crop height is below target_height
        h, w = crop.shape[:2]
        if 0 < h < target_height:
            scale_fac = target_height / float(h)
            crop = cls.scale_for_ocr(crop, scale_factor=scale_fac)

        res_type = resource_type.lower()
        strat = effective_strat

        if strat == "auto":
            if res_type == "dark_elixir":
                strat = "luminance_mask"
            elif res_type in ("gold", "elixir"):
                strat = "clahe_otsu"
            else:
                strat = "clahe_otsu"

        if strat == "luminance_mask":
            # Isolate bright digits (V >= 190, S <= 70 in HSV)
            if crop.ndim == 3 and crop.shape[2] >= 3:
                hsv = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 70, 255]))
                if cv2.countNonZero(mask) > 10:
                    return cv2.bitwise_not(mask)
            strat = "clahe_otsu"

        if strat == "bilateral_adaptive":
            gray = cls.to_grayscale(crop)
            bilateral = cv2.bilateralFilter(gray, 5, 50, 50)
            thresh = cls.adaptive_threshold(bilateral, method="gaussian", block_size=15, c=3, invert=True)
            return cls.remove_noise(thresh, kernel_size=1, operation="open")

        if strat == "invert_thresh":
            gray = cls.to_grayscale(crop)
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            return thresh

        if strat == "adaptive_gaussian":
            gray = cls.to_grayscale(crop)
            enhanced = cls.enhance_contrast(gray, clip_limit=3.0)
            thresh = cls.adaptive_threshold(enhanced, method="gaussian", block_size=15, c=3, invert=True)
            return cls.remove_noise(thresh, kernel_size=1, operation="open")

        # Default clahe_otsu
        gray = cls.to_grayscale(crop)
        enhanced = cls.enhance_contrast(gray, clip_limit=2.5)
        thresh = cls.adaptive_threshold(enhanced, method="otsu", invert=True)
        return cls.remove_noise(thresh, kernel_size=1, operation="open")

    @classmethod
    def match_template_multiscale(
        cls,
        image: Optional[np.ndarray],
        template: Optional[np.ndarray],
        scales: Optional[Union[List[float], Tuple[float, ...]]] = None,
        method: str = "grayscale",
        match_method: int = cv2.TM_CCOEFF_NORMED,
        threshold: float = 0.70,
        early_exit_threshold: Optional[float] = None,
        canny_low: int = 50,
        canny_high: int = 150,
    ) -> TemplateMatchResult:
        """
        Resolution-independent multi-scale template matching across user-defined scaling levels.
        
        Args:
            image: Source scene image (BGR, BGRA, or grayscale).
            template: Target template image to detect.
            scales: Sequence of float scales (e.g. [0.75, 1.0, 1.25]) or None for default [0.65 .. 1.35].
            method: 'grayscale' for luminance correlation, or 'edge' / 'canny' for structural edge matching.
            match_method: OpenCV template matching method (default: cv2.TM_CCOEFF_NORMED).
            threshold: Minimum correlation score [0.0, 1.0] to declare a positive match.
            early_exit_threshold: Optional correlation score (e.g. 0.95) that terminates search immediately upon match.
            canny_low: Lower hysteresis threshold for Canny edge detector (used when method='edge').
            canny_high: Upper hysteresis threshold for Canny edge detector (used when method='edge').
            
        Returns:
            TemplateMatchResult containing match coordinates, confidence score, and optimal scale.
        """
        method_norm = method.lower()
        effective_method = "edge" if method_norm in ("edge", "canny") else "grayscale"

        # 1. Validation & boundary checks
        if image is None or template is None:
            return TemplateMatchResult(
                found=False, confidence=0.0, x=0, y=0, width=0, height=0, scale=1.0, method=effective_method
            )

        if image.size == 0 or template.size == 0 or image.ndim < 2 or template.ndim < 2:
            return TemplateMatchResult(
                found=False, confidence=0.0, x=0, y=0, width=0, height=0, scale=1.0, method=effective_method
            )

        # 2. Standardize grayscale
        gray_img = cls.to_grayscale(image)
        gray_tpl = cls.to_grayscale(template)

        if gray_img.size == 0 or gray_tpl.size == 0:
            return TemplateMatchResult(
                found=False, confidence=0.0, x=0, y=0, width=0, height=0, scale=1.0, method=effective_method
            )

        img_h, img_w = gray_img.shape[:2]
        tpl_h, tpl_w = gray_tpl.shape[:2]

        # 3. Optional edge detection
        use_edge = (effective_method == "edge")
        if use_edge:
            edge_img = cv2.Canny(gray_img, canny_low, canny_high)
            if cv2.countNonZero(edge_img) == 0:
                # Flat scene, fallback to grayscale
                use_edge = False
                effective_method = "grayscale"
            else:
                gray_img = edge_img

        # 4. Prepare scale sequence
        if scales is None:
            scale_list = [round(float(s), 2) for s in np.arange(0.65, 1.36, 0.05)]
        else:
            scale_list = [float(s) for s in scales if s > 0]

        if not scale_list:
            scale_list = [1.0]

        best_score = -1.0 if match_method not in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED) else float("inf")
        best_loc = (0, 0)
        best_scale = 1.0
        best_w, best_h = 0, 0

        # 5. Multi-scale search loop
        for s in scale_list:
            target_w = int(round(tpl_w * s))
            target_h = int(round(tpl_h * s))

            # Discard impossible template sizes
            if target_w < 5 or target_h < 5:
                continue
            if target_w > img_w or target_h > img_h:
                continue

            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
            scaled_tpl = cv2.resize(gray_tpl, (target_w, target_h), interpolation=interp)

            if use_edge:
                scaled_tpl = cv2.Canny(scaled_tpl, canny_low, canny_high)
                if cv2.countNonZero(scaled_tpl) == 0:
                    continue

            try:
                res = cv2.matchTemplate(gray_img, scaled_tpl, match_method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if match_method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                    score = 1.0 - min_val if match_method == cv2.TM_SQDIFF_NORMED else -min_val
                    loc = min_loc
                else:
                    score = float(max_val)
                    loc = max_loc

                if score > best_score:
                    best_score = score
                    best_loc = loc
                    best_scale = s
                    best_w = target_w
                    best_h = target_h

                if early_exit_threshold is not None and score >= early_exit_threshold:
                    break
            except cv2.error:
                continue

        # 6. Build typed result
        raw_conf = best_score if best_w > 0 else 0.0
        conf = float(max(0.0, min(1.0, raw_conf)))
        found = bool(conf >= threshold and best_w > 0)

        return TemplateMatchResult(
            found=found,
            confidence=conf,
            x=int(best_loc[0]),
            y=int(best_loc[1]),
            width=int(best_w),
            height=int(best_h),
            scale=float(best_scale),
            method=effective_method,
        )

    @classmethod
    def match_template(
        cls,
        image: Optional[np.ndarray],
        template: Optional[np.ndarray],
        method: str = "grayscale",
        match_method: int = cv2.TM_CCOEFF_NORMED,
        threshold: float = 0.70,
    ) -> TemplateMatchResult:
        """
        Fast single-scale (1.0x) template matching convenience wrapper.
        """
        return cls.match_template_multiscale(
            image=image,
            template=template,
            scales=[1.0],
            method=method,
            match_method=match_method,
            threshold=threshold,
        )
