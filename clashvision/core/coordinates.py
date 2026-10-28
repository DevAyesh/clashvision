"""
ClashVision coordinate transformations, DPI normalization, and geometry utilities.
Enables resolution-independent spatial reasoning across arbitrary client window sizes,
virtual reference resolutions, and multi-monitor DPI environments.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional, Union
from clashvision.core.constants import (
    VIRTUAL_REFERENCE_WIDTH,
    VIRTUAL_REFERENCE_HEIGHT,
    REFERENCE_ASPECT_RATIO,
)
from clashvision.core.models import (
    Point,
    NormalizedPoint,
    Rect,
    NormalizedRect,
)


def point_to_normalized(
    pt: Union[Point, Tuple[int, int]],
    width: int,
    height: int,
) -> NormalizedPoint:
    """Converts a pixel point into normalized [0.0, 1.0] coordinate space."""
    if width <= 0 or height <= 0:
        return NormalizedPoint(0.0, 0.0)
    px = pt.x if isinstance(pt, Point) else pt[0]
    py = pt.y if isinstance(pt, Point) else pt[1]
    if math.isnan(px) or math.isnan(py):
        return NormalizedPoint(1.0, 1.0)
    if math.isinf(px) or math.isinf(py):
        nx = 1.0 if px > 0 else 0.0
        ny = 1.0 if py > 0 else 0.0
        return NormalizedPoint(nx, ny)
    nx = max(0.0, min(1.0, float(px) / float(width)))
    ny = max(0.0, min(1.0, float(py) / float(height)))
    return NormalizedPoint(nx, ny)


def normalized_to_point(
    npt: Union[NormalizedPoint, Tuple[float, float]],
    width: int,
    height: int,
) -> Point:
    """Converts a normalized [0.0, 1.0] coordinate into pixel space."""
    if width <= 0 or height <= 0:
        return Point(0, 0)
    nx = npt.x if isinstance(npt, NormalizedPoint) else npt[0]
    ny = npt.y if isinstance(npt, NormalizedPoint) else npt[1]
    if math.isnan(nx) or math.isnan(ny):
        return Point(width, height)
    if math.isinf(nx) or math.isinf(ny):
        px = width if nx > 0 else 0
        py = height if ny > 0 else 0
        return Point(px, py)
    px = int(round(max(0.0, min(1.0, nx)) * width))
    py = int(round(max(0.0, min(1.0, ny)) * height))
    return Point(px, py)


def rect_to_normalized(
    rect: Rect,
    width: int,
    height: int,
) -> NormalizedRect:
    """Converts a pixel bounding box into normalized [0.0, 1.0] space."""
    if width <= 0 or height <= 0:
        return NormalizedRect(0.0, 0.0, 0.0, 0.0)
    nx = max(0.0, min(1.0, float(rect.x) / float(width)))
    ny = max(0.0, min(1.0, float(rect.y) / float(height)))
    nw = max(0.0, min(1.0 - nx, float(rect.width) / float(width)))
    nh = max(0.0, min(1.0 - ny, float(rect.height) / float(height)))
    return NormalizedRect(nx, ny, nw, nh)


def normalized_to_rect(
    nrect: NormalizedRect,
    width: int,
    height: int,
) -> Rect:
    """Converts a normalized [0.0, 1.0] bounding box into pixel space."""
    if width <= 0 or height <= 0:
        return Rect(0, 0, 0, 0)
    x = int(round(max(0.0, min(1.0, nrect.x)) * width))
    y = int(round(max(0.0, min(1.0, nrect.y)) * height))
    w = int(round(max(0.0, min(1.0, nrect.width)) * width))
    h = int(round(max(0.0, min(1.0, nrect.height)) * height))
    return clamp_rect_to_bounds(Rect(x, y, w, h), width, height)


def clamp_point_to_bounds(
    pt: Union[Point, Tuple[int, int]],
    width: int,
    height: int,
) -> Point:
    """Clamps a pixel coordinate strictly within [0, width - 1] and [0, height - 1]."""
    if width <= 0 or height <= 0:
        return Point(0, 0)
    px = pt.x if isinstance(pt, Point) else pt[0]
    py = pt.y if isinstance(pt, Point) else pt[1]
    return Point(
        x=max(0, min(width - 1, px)),
        y=max(0, min(height - 1, py)),
    )


def clamp_rect_to_bounds(
    rect: Rect,
    width: int,
    height: int,
) -> Rect:
    """Clamps a rectangle so it resides entirely inside the canvas bounds [0, 0, width, height]."""
    if width <= 0 or height <= 0:
        return Rect(0, 0, 0, 0)
    x1 = max(0, min(width, rect.x))
    y1 = max(0, min(height, rect.y))
    x2 = max(x1, min(width, rect.x + rect.width))
    y2 = max(y1, min(height, rect.y + rect.height))
    return Rect(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def scale_rect_to_aspect(
    rect: Rect,
    target_aspect: float = REFERENCE_ASPECT_RATIO,
) -> Rect:
    """
    Fits an arbitrary bounding box to match a target aspect ratio
    (center-aligned) while keeping it within bounds.
    """
    if math.isnan(target_aspect):
        raise ValueError("target_aspect cannot be NaN")
    if math.isinf(target_aspect):
        raise ValueError("target_aspect cannot be infinite")
    if not rect.is_valid or target_aspect <= 0:
        return rect
    current_aspect = rect.width / rect.height
    if current_aspect > target_aspect:
        # Too wide -> shrink width
        new_width = int(round(rect.height * target_aspect))
        new_x = rect.x + (rect.width - new_width) // 2
        return Rect(new_x, rect.y, new_width, rect.height)
    else:
        # Too tall -> shrink height
        new_height = int(round(rect.width / target_aspect))
        new_y = rect.y + (rect.height - new_height) // 2
        return Rect(rect.x, new_y, rect.width, new_height)


def calculate_letterbox(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> Tuple[Rect, float]:
    """
    Calculates letterbox/pillarbox viewport and uniform scale factor when fitting
    a source canvas (e.g. 1920x1080) into a target display window.
    Returns (content_rect, uniform_scale).
    """
    if any(math.isnan(v) for v in (source_width, source_height, target_width, target_height)):
        raise ValueError("calculate_letterbox dimensions cannot be NaN")
    if any(math.isinf(v) for v in (source_width, source_height, target_width, target_height)):
        raise OverflowError("calculate_letterbox dimensions cannot be infinite")
    if source_width <= 0 or source_height <= 0 or target_width <= 0 or target_height <= 0:
        return (Rect(0, 0, 0, 0), 1.0)

    scale_x = target_width / source_width
    scale_y = target_height / source_height
    scale = min(scale_x, scale_y)

    content_w = int(round(source_width * scale))
    content_h = int(round(source_height * scale))
    offset_x = (target_width - content_w) // 2
    offset_y = (target_height - content_h) // 2

    return (Rect(offset_x, offset_y, content_w, content_h), scale)


def calculate_iou(r1: Rect, r2: Rect) -> float:
    """
    Computes Intersection over Union (IoU) between two rectangles.
    Returns value in [0.0, 1.0].
    """
    if not r1.is_valid or not r2.is_valid:
        return 0.0

    inter_x1 = max(r1.left, r2.left)
    inter_y1 = max(r1.top, r2.top)
    inter_x2 = min(r1.right, r2.right)
    inter_y2 = min(r1.bottom, r2.bottom)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    union_area = r1.area + r2.area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def intersect_rects(r1: Rect, r2: Rect) -> Optional[Rect]:
    """Returns the intersection of two rectangles, or None if disjoint."""
    inter_x1 = max(r1.left, r2.left)
    inter_y1 = max(r1.top, r2.top)
    inter_x2 = min(r1.right, r2.right)
    inter_y2 = min(r1.bottom, r2.bottom)

    w = inter_x2 - inter_x1
    h = inter_y2 - inter_y1
    if w <= 0 or h <= 0:
        return None
    return Rect(inter_x1, inter_y1, w, h)


def union_rects(r1: Rect, r2: Rect) -> Rect:
    """Returns the minimum bounding rectangle encompassing both rectangles."""
    if not r1.is_valid:
        return r2
    if not r2.is_valid:
        return r1
    min_x = min(r1.left, r2.left)
    min_y = min(r1.top, r2.top)
    max_x = max(r1.right, r2.right)
    max_y = max(r1.bottom, r2.bottom)
    return Rect(min_x, min_y, max_x - min_x, max_y - min_y)


class CoordinateTransformer:
    """
    Resolution and DPI transformer facilitating conversions between:
    - Virtual reference canvas (e.g. 1920x1080)
    - Client window coordinates
    - Normalized [0.0, 1.0] coordinates
    - Physical screen coordinates
    """

    def __init__(
        self,
        client_width: Optional[int] = None,
        client_height: Optional[int] = None,
        virtual_width: int = VIRTUAL_REFERENCE_WIDTH,
        virtual_height: int = VIRTUAL_REFERENCE_HEIGHT,
        scale_factor: float = 1.0,
        dpi: int = 96,
        client_rect: Optional[Rect] = None,
    ):
        if client_rect is not None:
            c_w = client_rect.width
            c_h = client_rect.height
        else:
            c_w = client_width if client_width is not None else VIRTUAL_REFERENCE_WIDTH
            c_h = client_height if client_height is not None else VIRTUAL_REFERENCE_HEIGHT

        self.client_width = max(1, c_w)
        self.client_height = max(1, c_h)
        self.virtual_width = max(1, virtual_width)
        self.virtual_height = max(1, virtual_height)
        self.scale_factor = scale_factor if scale_factor > 0 else 1.0
        self.dpi = dpi if dpi > 0 else 96

    def update_client_size(self, width: int, height: int, scale_factor: float = 1.0, dpi: int = 96) -> None:
        """Updates client dimensions dynamically on window resize."""
        self.client_width = max(1, width)
        self.client_height = max(1, height)
        self.scale_factor = scale_factor if scale_factor > 0 else 1.0
        self.dpi = dpi if dpi > 0 else 96

    # 1. Virtual <-> Client Transforms
    def virtual_to_client(self, target: Union[Point, Rect]) -> Union[Point, Rect]:
        """Convenience dispatcher for mapping virtual points or rects to client space."""
        if isinstance(target, Rect):
            return self.virtual_to_client_rect(target)
        return self.virtual_to_client_point(target)

    def client_to_virtual(self, target: Union[Point, Rect]) -> Union[Point, Rect]:
        """Convenience dispatcher for mapping client points or rects to virtual space."""
        if isinstance(target, Rect):
            return self.client_to_virtual_rect(target)
        return self.client_to_virtual_point(target)

    def virtual_to_client_point(self, pt: Point) -> Point:
        """Maps virtual reference coordinate (e.g. 1920x1080) to client window pixel coordinate."""
        if math.isnan(pt.x) or math.isnan(pt.y):
            raise ValueError("Point coordinates cannot be NaN")
        if math.isinf(pt.x) or math.isinf(pt.y):
            raise OverflowError("Point coordinates cannot be Inf")
        nx = pt.x / self.virtual_width
        ny = pt.y / self.virtual_height
        cx = int(round(nx * self.client_width))
        cy = int(round(ny * self.client_height))
        return Point(cx, cy)

    def client_to_virtual_point(self, pt: Point) -> Point:
        """Maps client window pixel coordinate to virtual reference coordinate (1920x1080)."""
        if math.isnan(pt.x) or math.isnan(pt.y):
            raise ValueError("Point coordinates cannot be NaN")
        if math.isinf(pt.x) or math.isinf(pt.y):
            raise OverflowError("Point coordinates cannot be Inf")
        nx = pt.x / self.client_width
        ny = pt.y / self.client_height
        vx = int(round(nx * self.virtual_width))
        vy = int(round(ny * self.virtual_height))
        return Point(vx, vy)

    def virtual_to_client_rect(self, rect: Rect) -> Rect:
        """Scales a virtual ROI bounding box to client space."""
        if any(math.isnan(v) for v in (rect.x, rect.y, rect.width, rect.height)):
            raise ValueError("Rect coordinates cannot be NaN")
        if any(math.isinf(v) for v in (rect.x, rect.y, rect.width, rect.height)):
            raise OverflowError("Rect coordinates cannot be Inf")
        nx = rect.x / self.virtual_width
        ny = rect.y / self.virtual_height
        nw = rect.width / self.virtual_width
        nh = rect.height / self.virtual_height

        cx = int(round(nx * self.client_width))
        cy = int(round(ny * self.client_height))
        cw = int(round(nw * self.client_width))
        ch = int(round(nh * self.client_height))
        return Rect(cx, cy, cw, ch)

    def client_to_virtual_rect(self, rect: Rect) -> Rect:
        """Scales a client ROI bounding box to virtual space."""
        if any(math.isnan(v) for v in (rect.x, rect.y, rect.width, rect.height)):
            raise ValueError("Rect coordinates cannot be NaN")
        if any(math.isinf(v) for v in (rect.x, rect.y, rect.width, rect.height)):
            raise OverflowError("Rect coordinates cannot be Inf")
        nx = rect.x / self.client_width
        ny = rect.y / self.client_height
        nw = rect.width / self.client_width
        nh = rect.height / self.client_height

        vx = int(round(nx * self.virtual_width))
        vy = int(round(ny * self.virtual_height))
        vw = int(round(nw * self.virtual_width))
        vh = int(round(nh * self.virtual_height))
        return Rect(vx, vy, vw, vh)

    # 2. Normalized Transforms
    def client_to_normalized_point(self, pt: Point) -> NormalizedPoint:
        return point_to_normalized(pt, self.client_width, self.client_height)

    def normalized_to_client_point(self, npt: NormalizedPoint) -> Point:
        return normalized_to_point(npt, self.client_width, self.client_height)

    def client_to_normalized_rect(self, rect: Rect) -> NormalizedRect:
        return rect_to_normalized(rect, self.client_width, self.client_height)

    def normalized_to_client_rect(self, nrect: NormalizedRect) -> Rect:
        return normalized_to_rect(nrect, self.client_width, self.client_height)

    # 3. Screen <-> Client (taking window origin into account)
    @staticmethod
    def client_to_screen_point(client_pt: Point, client_origin: Point) -> Point:
        """Converts client relative coordinate to absolute desktop screen coordinate."""
        return Point(client_pt.x + client_origin.x, client_pt.y + client_origin.y)

    @staticmethod
    def screen_to_client_point(screen_pt: Point, client_origin: Point) -> Point:
        """Converts absolute desktop screen coordinate to client relative coordinate."""
        return Point(screen_pt.x - client_origin.x, screen_pt.y - client_origin.y)

    # 4. DPI awareness transforms
    @staticmethod
    def apply_dpi_scale(val: float, dpi: int = 96) -> float:
        """Applies DPI scaling factor relative to 96 DPI baseline."""
        factor = max(0.1, dpi / 96.0)
        return val * factor

    @staticmethod
    def remove_dpi_scale(val: float, dpi: int = 96) -> float:
        """Removes DPI scaling factor to recover standard 96 DPI coordinate."""
        factor = max(0.1, dpi / 96.0)
        return val / factor
