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
