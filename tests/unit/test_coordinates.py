"""
Unit tests for coordinate transformations, DPI scaling, and geometry utilities.
Verifies Feature 3 (DPI Scaling & Dynamic Resize Transform) and related requirements.
"""

import unittest
import pytest
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
from clashvision.core.coordinates import (
    CoordinateTransformer,
    point_to_normalized,
    normalized_to_point,
    rect_to_normalized,
    normalized_to_rect,
    clamp_point_to_bounds,
    clamp_rect_to_bounds,
    scale_rect_to_aspect,
    calculate_letterbox,
    calculate_iou,
    intersect_rects,
    union_rects,
)


class TestGeometryModels(unittest.TestCase):
    """Tests for pure dataclass geometry primitives (Point, Rect, etc.)."""

    def test_point_distance_and_tuple(self):
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        assert p1.to_tuple() == (0, 0)
        assert p2.to_tuple() == (3, 4)
        assert p1.distance_to(p2) == 5.0
        assert p2.distance_to(p1) == 5.0

    def test_normalized_point_conversion(self):
        np_pt = NormalizedPoint(0.5, 0.25)
        pixel_pt = np_pt.to_pixel(1920, 1080)
        assert pixel_pt.x == 960
        assert pixel_pt.y == 270

    def test_rect_properties(self):
        r = Rect(100, 200, 300, 400)
        assert r.left == 100
        assert r.top == 200
        assert r.right == 400
        assert r.bottom == 600
        assert r.center_x == 250
        assert r.center_y == 400
        assert r.area == 120000
        assert r.is_valid is True

    def test_rect_contains_point(self):
        r = Rect(50, 50, 100, 100)
        assert r.contains_point(Point(50, 50)) is True
        assert r.contains_point((100, 100)) is True
        assert r.contains_point(Point(149, 149)) is True
        assert r.contains_point(Point(150, 150)) is False
        assert r.contains_point(Point(49, 50)) is False

    def test_rect_intersects(self):
        r1 = Rect(0, 0, 100, 100)
        r2 = Rect(50, 50, 100, 100)
        r3 = Rect(100, 100, 50, 50)
        r4 = Rect(200, 200, 50, 50)

        assert r1.intersects(r2) is True
        assert r2.intersects(r1) is True
        assert r1.intersects(r3) is False  # Touching edge is not intersecting
        assert r1.intersects(r4) is False

    def test_rect_serialization(self):
        r = Rect(10, 20, 30, 40)
        d = r.to_dict()
        assert d == {"x": 10, "y": 20, "width": 30, "height": 40}
        r2 = Rect.from_dict(d)
        assert r == r2
        assert r.to_tuple() == (10, 20, 30, 40)

    def test_normalized_rect_to_pixel(self):
        nr = NormalizedRect(0.1, 0.2, 0.5, 0.4)
        pr = nr.to_pixel_rect(1000, 1000)
        assert pr.x == 100
        assert pr.y == 200
        assert pr.width == 500
        assert pr.height == 400


class TestNormalizationFunctions(unittest.TestCase):
    """Tests for normalized <-> pixel space conversions."""

    def test_point_to_normalized_nominal(self):
        npt = point_to_normalized(Point(960, 540), 1920, 1080)
        assert pytest.approx(npt.x, 0.001) == 0.5
        assert pytest.approx(npt.y, 0.001) == 0.5

    def test_point_to_normalized_clamping(self):
        # Negative point
        npt_neg = point_to_normalized(Point(-50, -20), 1920, 1080)
        assert npt_neg.x == 0.0
        assert npt_neg.y == 0.0

        # Out-of-bounds point
        npt_oob = point_to_normalized(Point(2500, 1500), 1920, 1080)
        assert npt_oob.x == 1.0
        assert npt_oob.y == 1.0

    def test_point_to_normalized_zero_dimension(self):
        npt = point_to_normalized(Point(100, 100), 0, 0)
        assert npt.x == 0.0
        assert npt.y == 0.0

    def test_normalized_to_point_nominal(self):
        pt = normalized_to_point(NormalizedPoint(0.5, 0.5), 1920, 1080)
        assert pt.x == 960
        assert pt.y == 540

    def test_normalized_to_point_zero_dimension(self):
        pt = normalized_to_point(NormalizedPoint(0.5, 0.5), 0, 0)
        assert pt.x == 0
        assert pt.y == 0

    def test_rect_to_normalized_roundtrip(self):
        orig_rect = Rect(192, 108, 960, 540)
        nrect = rect_to_normalized(orig_rect, 1920, 1080)
        assert pytest.approx(nrect.x, 0.001) == 0.1
        assert pytest.approx(nrect.y, 0.001) == 0.1
        assert pytest.approx(nrect.width, 0.001) == 0.5
        assert pytest.approx(nrect.height, 0.001) == 0.5

        reconstructed = normalized_to_rect(nrect, 1920, 1080)
        assert reconstructed.x == orig_rect.x
        assert reconstructed.y == orig_rect.y
        assert reconstructed.width == orig_rect.width
        assert reconstructed.height == orig_rect.height


class TestGeometryUtilities(unittest.TestCase):
    """Tests for clamping, aspect scaling, IoU, letterboxing."""

    def test_clamp_point_to_bounds(self):
        p1 = clamp_point_to_bounds(Point(-10, 500), 1920, 1080)
        assert p1.x == 0
        assert p1.y == 500

        p2 = clamp_point_to_bounds(Point(2000, 1500), 1920, 1080)
        assert p2.x == 1919
        assert p2.y == 1079

    def test_clamp_rect_to_bounds(self):
        # Partly outside left/top
        r1 = clamp_rect_to_bounds(Rect(-50, -50, 200, 200), 1000, 1000)
        assert r1.x == 0
        assert r1.y == 0
        assert r1.width == 150
        assert r1.height == 150

        # Partly outside right/bottom
        r2 = clamp_rect_to_bounds(Rect(900, 900, 300, 300), 1000, 1000)
        assert r2.x == 900
        assert r2.y == 900
        assert r2.width == 100
        assert r2.height == 100

        # Completely outside
        r3 = clamp_rect_to_bounds(Rect(1200, 1200, 200, 200), 1000, 1000)
        assert r3.width == 0
        assert r3.height == 0

    def test_scale_rect_to_aspect(self):
        # 16:9 target aspect = 1.7777777778
        # Rect with 2:1 aspect (too wide)
        r_wide = Rect(0, 0, 200, 100)
        fitted_w = scale_rect_to_aspect(r_wide, 16.0 / 9.0)
        assert fitted_w.height == 100
        assert fitted_w.width == 178  # int(round(100 * 16 / 9))
        assert fitted_w.x == (200 - 178) // 2

        # Rect with 1:1 aspect (too tall)
        r_tall = Rect(0, 0, 100, 100)
        fitted_t = scale_rect_to_aspect(r_tall, 16.0 / 9.0)
        assert fitted_t.width == 100
        assert fitted_t.height == 56  # int(round(100 / (16/9)))
        assert fitted_t.y == (100 - 56) // 2

    def test_calculate_letterbox_same_aspect(self):
        # 1920x1080 into 1280x720 (both 16:9)
        viewport, scale = calculate_letterbox(1920, 1080, 1280, 720)
        assert pytest.approx(scale, 0.001) == (1280 / 1920)
        assert viewport.x == 0
        assert viewport.y == 0
        assert viewport.width == 1280
        assert viewport.height == 720

    def test_calculate_letterbox_pillarbox(self):
        # 16:9 source (1920x1080) into 4:3 target (1600x1200)
        viewport, scale = calculate_letterbox(1920, 1080, 1600, 1200)
        # scale should be bounded by width: 1600 / 1920 = 0.8333
        assert pytest.approx(scale, 0.001) == (1600 / 1920)
        assert viewport.width == 1600
        assert viewport.height == int(round(1080 * (1600 / 1920)))  # 900
        assert viewport.x == 0
        assert viewport.y == (1200 - 900) // 2  # 150 (letterbox top/bottom bars)

    def test_calculate_iou(self):
        r1 = Rect(0, 0, 100, 100)
        r2 = Rect(0, 0, 100, 100)
        # Identical rects -> IoU = 1.0
        assert pytest.approx(calculate_iou(r1, r2), 0.001) == 1.0

        # Disjoint rects -> IoU = 0.0
        r3 = Rect(200, 200, 50, 50)
        assert calculate_iou(r1, r3) == 0.0

        # Half-overlapping rects: overlap is 50x100 = 5000, union = 10000 + 10000 - 5000 = 15000 -> IoU = 1/3
        r4 = Rect(50, 0, 100, 100)
        assert pytest.approx(calculate_iou(r1, r4), 0.001) == (5000.0 / 15000.0)

        # Invalid rects
        r_empty = Rect(0, 0, 0, 0)
        assert calculate_iou(r1, r_empty) == 0.0

    def test_intersect_and_union_rects(self):
        r1 = Rect(0, 0, 100, 100)
        r2 = Rect(50, 50, 100, 100)

        intersection = intersect_rects(r1, r2)
        assert intersection is not None
        assert intersection == Rect(50, 50, 50, 50)

        union = union_rects(r1, r2)
        assert union == Rect(0, 0, 150, 150)

        # Disjoint intersection
        r3 = Rect(200, 200, 50, 50)
        assert intersect_rects(r1, r3) is None


class TestCoordinateTransformer(unittest.TestCase):
    """Tests for multi-resolution and DPI transformer class."""

    def test_virtual_to_client_identity(self):
        transformer = CoordinateTransformer(1920, 1080, 1920, 1080)
        pt = Point(500, 300)
        assert transformer.virtual_to_client_point(pt) == pt
        assert transformer.client_to_virtual_point(pt) == pt

    def test_virtual_to_client_downscale(self):
        # 1920x1080 -> 960x540 (0.5x scale)
        transformer = CoordinateTransformer(960, 540, 1920, 1080)
        v_pt = Point(1000, 600)
        c_pt = transformer.virtual_to_client_point(v_pt)
        assert c_pt.x == 500
        assert c_pt.y == 300

        # Back to virtual
        reconstructed = transformer.client_to_virtual_point(c_pt)
        assert reconstructed.x == 1000
        assert reconstructed.y == 600

    def test_virtual_to_client_rect_scaling(self):
        transformer = CoordinateTransformer(1280, 720, 1920, 1080)
        v_rect = Rect(192, 108, 960, 540)
        c_rect = transformer.virtual_to_client_rect(v_rect)
        assert c_rect.x == 128
        assert c_rect.y == 72
        assert c_rect.width == 640
        assert c_rect.height == 360

        v_reconstructed = transformer.client_to_virtual_rect(c_rect)
        assert v_reconstructed.x == 192
        assert v_reconstructed.y == 108
        assert v_reconstructed.width == 960
        assert v_reconstructed.height == 540

    def test_client_to_screen_offset(self):
        client_origin = Point(250, 150)
        client_pt = Point(50, 50)
        screen_pt = CoordinateTransformer.client_to_screen_point(client_pt, client_origin)
        assert screen_pt.x == 300
        assert screen_pt.y == 200

        back_to_client = CoordinateTransformer.screen_to_client_point(screen_pt, client_origin)
        assert back_to_client == client_pt

    def test_dpi_scaling(self):
        # 96 DPI baseline -> scale factor 1.0
        assert CoordinateTransformer.apply_dpi_scale(100, 96) == 100.0
        assert CoordinateTransformer.remove_dpi_scale(100, 96) == 100.0

        # 144 DPI (150% scaling)
        scaled_150 = CoordinateTransformer.apply_dpi_scale(100, 144)
        assert pytest.approx(scaled_150, 0.001) == 150.0
        unscaled_150 = CoordinateTransformer.remove_dpi_scale(scaled_150, 144)
        assert pytest.approx(unscaled_150, 0.001) == 100.0

        # 192 DPI (200% scaling)
        scaled_200 = CoordinateTransformer.apply_dpi_scale(50, 192)
        assert pytest.approx(scaled_200, 0.001) == 100.0
        assert pytest.approx(CoordinateTransformer.remove_dpi_scale(100, 192), 0.001) == 50.0

    def test_dynamic_client_resize(self):
        transformer = CoordinateTransformer(1920, 1080)
        assert transformer.client_width == 1920

        # Resize to 2560x1440
        transformer.update_client_size(2560, 1440, scale_factor=1.25, dpi=120)
        assert transformer.client_width == 2560
        assert transformer.client_height == 1440
        assert transformer.scale_factor == 1.25
        assert transformer.dpi == 120

        v_pt = Point(960, 540)
        c_pt = transformer.virtual_to_client_point(v_pt)
        assert c_pt.x == 1280
        assert c_pt.y == 720
