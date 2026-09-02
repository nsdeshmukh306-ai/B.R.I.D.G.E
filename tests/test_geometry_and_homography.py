import numpy as np
import pytest

from bridge.spatial.geometry import BoundingBox, Point, Polygon
from bridge.spatial.homography import CoordinateMapper, HomographyError, Matrix3x3, apply_homography, compute_homography


def test_bbox_from_normalized_gemini_order():
    # Gemini: [ymin, xmin, ymax, xmax] on 0..1000
    b = BoundingBox.from_normalized([100, 200, 300, 600], width=1000, height=500, scale=1000.0)
    assert (b.x, b.y, b.w, b.h) == (200.0, 50.0, 400.0, 100.0)
    assert b.center == Point(x=400.0, y=100.0)


def test_bbox_from_normalized_unit_scale_and_clip():
    b = BoundingBox.from_normalized([0.5, 0.5, 1.2, 1.5], width=100, height=100, order="ymin_xmin_ymax_xmax", scale=1.0)
    assert b.x2 <= 100 and b.y2 <= 100


def test_bbox_iou_and_polygon_contains():
    a = BoundingBox(x=0, y=0, w=10, h=10)
    b = BoundingBox(x=5, y=5, w=10, h=10)
    assert abs(a.iou(b) - 25 / 175) < 1e-9
    poly = Polygon(points=[Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=10), Point(x=0, y=10)])
    assert poly.contains(Point(x=5, y=5)) and not poly.contains(Point(x=15, y=5))


def test_homography_known_four_corners():
    src = [Point(x=0, y=0), Point(x=640, y=0), Point(x=640, y=480), Point(x=0, y=480)]
    dst = [Point(x=100, y=50), Point(x=1800, y=80), Point(x=1900, y=1000), Point(x=50, y=1050)]
    H, mask = compute_homography(src, dst)
    mapped = apply_homography(H, src)
    assert np.allclose(mapped, [[p.x, p.y] for p in dst], atol=1e-3)
    assert mask.sum() == 4


def test_coordinate_mapper_roundtrip():
    H = np.array([[1.2, 0.1, 30], [0.05, 0.9, -20], [1e-4, 2e-4, 1]])
    m = CoordinateMapper(H)
    p = Point(x=320, y=240)
    q = m.projector_to_camera(m.camera_to_projector(p))
    assert p.distance_to(q) < 1e-6
    poly = m.camera_bbox_to_projector_polygon(BoundingBox(x=10, y=10, w=50, h=30))
    assert len(poly.points) == 4
    assert m.camera_radius_to_projector(p, 10) > 0


def test_known_camera_point_maps_to_expected_projector_point():
    # Pure scale+translation: (x, y) -> (2x + 100, 3y + 50)
    H = np.array([[2, 0, 100], [0, 3, 50], [0, 0, 1]], dtype=float)
    m = CoordinateMapper(H)
    assert m.camera_to_projector(Point(x=10, y=20)) == Point(x=120.0, y=110.0)


def test_homography_rejects_degenerate_input():
    with pytest.raises(HomographyError):
        compute_homography([Point(x=0, y=0)] * 4, [Point(x=1, y=1)] * 4)
    with pytest.raises(HomographyError):
        compute_homography([Point(x=0, y=0)] * 3, [Point(x=1, y=1)] * 3)


def test_matrix3x3_roundtrip_and_validation():
    m = Matrix3x3.identity()
    assert np.allclose(m.to_numpy(), np.eye(3))
    with pytest.raises(ValueError):
        Matrix3x3(rows=[[1, 2], [3, 4]])
