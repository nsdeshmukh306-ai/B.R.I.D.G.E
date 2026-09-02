import numpy as np
import pytest

from bridge.spatial.geometry import BoundingBox, Point, Polygon
from bridge.spatial.homography import CoordinateMapper, HomographyError, Matrix3x3, apply_homography, compute_homography

SRC = [Point(x=0, y=0), Point(x=640, y=0), Point(x=640, y=480), Point(x=0, y=480)]
DST = [Point(x=100, y=50), Point(x=1800, y=80), Point(x=1900, y=1000), Point(x=50, y=1050)]


def test_known_four_corners_map_exactly():
    H, _ = compute_homography(SRC, DST)
    out = apply_homography(H, SRC)
    for got, want in zip(out, DST):
        assert np.allclose(got, [want.x, want.y], atol=1e-3)


def test_coordinate_mapper_round_trip():
    H, _ = compute_homography(SRC, DST)
    m = CoordinateMapper(H)
    p = Point(x=321, y=123)
    back = m.projector_to_camera(m.camera_to_projector(p))
    assert p.distance_to(back) < 1e-6


def test_known_camera_point_to_expected_projector_point():
    # pure scale+translation: x*2+100, y*3+50
    src = [Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=10), Point(x=0, y=10), Point(x=5, y=5)]
    dst = [Point(x=p.x * 2 + 100, y=p.y * 3 + 50) for p in src]
    H, _ = compute_homography(src, dst)
    m = CoordinateMapper(H)
    got = m.camera_to_projector(Point(x=7, y=3))
    assert abs(got.x - 114) < 1e-6 and abs(got.y - 59) < 1e-6


def test_degenerate_points_raise():
    pts = [Point(x=0, y=0), Point(x=1, y=1), Point(x=2, y=2), Point(x=3, y=3)]
    with pytest.raises(HomographyError):
        compute_homography(pts, DST)
    with pytest.raises(HomographyError):
        compute_homography(SRC[:3], DST[:3])


def test_matrix3x3_serialization_round_trip():
    H, _ = compute_homography(SRC, DST)
    m = Matrix3x3.from_numpy(H)
    assert np.allclose(Matrix3x3.model_validate(m.model_dump()).to_numpy(), H)


def test_normalized_bbox_conversion_gemini_format():
    b = BoundingBox.from_normalized([100, 200, 500, 600], width=1000, height=500, scale=1000.0)
    assert (b.x, b.y, b.w, b.h) == (200, 50, 400, 200)
    assert b.center == Point(x=400, y=150)


def test_normalized_bbox_clipped_and_xyxy_order():
    b = BoundingBox.from_normalized([0.5, 0.5, 1.5, 1.5], 100, 100, order="xmin_ymin_xmax_ymax")
    assert (b.x, b.y, b.x2, b.y2) == (50, 50, 100, 100)


def test_bbox_iou_and_polygon_contains():
    a = BoundingBox(x=0, y=0, w=10, h=10)
    b = BoundingBox(x=5, y=5, w=10, h=10)
    assert abs(a.iou(b) - 25 / 175) < 1e-9
    poly = Polygon(points=a.corners())
    assert poly.contains(Point(x=5, y=5)) and not poly.contains(Point(x=50, y=5))


def test_radius_mapping_scales_with_homography():
    H, _ = compute_homography(SRC, [p.scale(2) for p in SRC])
    m = CoordinateMapper(H)
    assert abs(m.camera_radius_to_projector(Point(x=100, y=100), 10) - 20) < 1e-6
