import numpy as np

from bridge.render.primitives import Style
from bridge.render.renderer import ProjectionRenderer, calibration_pattern
from bridge.spatial.geometry import BoundingBox, Point
from bridge.vision.detection import ContourDetector, Detection, describe_color, match_description
from bridge.vision.tracking import ObjectTracker, MultiTracker


def test_all_primitives_render_without_exceptions():
    r = ProjectionRenderer(640, 360, "white")
    r.circle(Point(x=300, y=180), 60, Style(animation="pulse"))
    r.circle(Point(x=100, y=100), 30, Style(dashed=True))
    r.outline([Point(x=10, y=10), Point(x=100, y=10), Point(x=100, y=80)])
    r.arrow(Point(x=10, y=10), Point(x=200, y=100), Style(animation="pulse"))
    r.point(Point(x=50, y=50))
    r.label(Point(x=300, y=100), "screwdriver")
    r.label(Point(x=-50, y=-50), "clipped label")  # off-canvas position is clamped
    r.target_zone([Point(x=400, y=200), Point(x=600, y=200), Point(x=600, y=340), Point(x=400, y=340)], Style(fill=True, opacity=0.4))
    r.path([Point(x=50, y=300), Point(x=200, y=250), Point(x=350, y=320)])
    r.message("Target lost.\nPlease clarify.")
    r.scene.add(r.scene.get(r.circle(Point(x=1, y=1), 5, Style(animation="blink"))))
    for t in (0.0, 0.3, 0.7):
        img = r.render(t)
        assert img.shape == (360, 640, 3) and img.dtype == np.uint8
    assert len(r.scene) == 11
    r.clear()
    assert len(r.scene) == 0 and np.all(r.render() == 255)


def test_background_modes():
    assert np.all(ProjectionRenderer(8, 8, "black").render() == 0)
    assert np.all(ProjectionRenderer(8, 8).render() == 0)  # black is the default canvas
    assert np.all(ProjectionRenderer(8, 8, "white").render() == 255)
    assert np.all(ProjectionRenderer(8, 8, "custom", (10, 20, 30)).render()[0, 0] == [30, 20, 10])


def test_render_mask_covers_only_graphics():
    r = ProjectionRenderer(200, 200)
    r.circle(Point(x=100, y=100), 40, Style(thickness=4, variant="ring", glow=False))
    m = r.render_mask()
    assert m[100, 60] == 255 and m[100, 100] == 0 and m[5, 5] == 0


def test_glow_and_reticle_render_on_black_and_stay_within_extent():
    r = ProjectionRenderer(400, 300)
    r.circle(Point(x=200, y=150), 50, Style(animation="pulse"))
    img = r.render(0.2)
    assert img[150, 145].max() > 150          # ring itself (pulse scales the radius slightly)
    assert 0 < img[150, 140].max() < 200      # soft glow just outside the ring
    assert img[10, 10].max() == 0             # far away stays black (projector emits nothing)
    r.background = "white"
    img_w = r.render(0.2)
    assert img_w[10, 10].min() == 255


def test_rounded_rect_and_label_pill():
    from bridge.render.renderer import rounded_rect_points

    pts = rounded_rect_points(10, 10, 100, 40, 12)
    assert pts.shape[1] == 2 and len(pts) == 28
    r = ProjectionRenderer(300, 100)
    r.label(Point(x=20, y=50), "syringe 5 mL")
    img = r.render()
    assert img[46:52, 30:60].max() > 100  # pill background is drawn around the text


def test_scene_groups():
    r = ProjectionRenderer(100, 100)
    r.circle(Point(x=10, y=10), 5, group="a")
    r.circle(Point(x=20, y=20), 5, group="a")
    r.point(Point(x=30, y=30), group="b")
    assert r.scene.remove_group("a") == 2 and len(r.scene) == 1


def test_calibration_pattern():
    img = calibration_pattern(100, 80, [Point(x=50, y=40)], radius=10)
    assert img[40, 50].tolist() == [255, 255, 255] and img[2, 2].tolist() == [0, 0, 0]


def test_color_description():
    assert describe_color((0, 0, 255)) == "red"
    assert describe_color((255, 0, 0)) == "blue"
    assert describe_color((20, 20, 20)) == "black"


def test_contour_detector_finds_simulated_objects(sim):
    world, proj, cam = sim
    dets = ContourDetector().detect(cam.render())
    assert len(dets) >= 5
    m = match_description(cam.render(), dets, "a red-handled screwdriver")
    gt = cam.object_bbox_in_camera(world.get("obj-screwdriver"))
    assert m is not None and m.center.distance_to(gt.center) < 15


def test_tracker_follows_moving_object_and_reports_loss(sim):
    world, proj, cam = sim
    frame = cam.render()
    gt = cam.object_bbox_in_camera(world.get("obj-screwdriver"))
    det = Detection.from_bbox("screwdriver", gt, 0.9, "test")
    tr = ObjectTracker(backend="color", lost_after_frames=5)
    tr.start(det, frame, color_hint="red")
    errs = []
    for _ in range(25):
        world.move_by("obj-screwdriver", 5, 3)
        s = tr.update(cam.render())
        gt = cam.object_bbox_in_camera(world.get("obj-screwdriver"))
        errs.append(s.center.distance_to(gt.center))
        assert not s.is_lost
    assert np.mean(errs) < 20
    world.remove("obj-screwdriver")
    for _ in range(8):
        s = tr.update(cam.render())
    assert s.is_lost
    tr.stop()
    assert not tr.active


def test_multitracker_tracks_all_screws(sim):
    world, proj, cam = sim
    frame = cam.render()
    dets = [Detection.from_bbox("screw", cam.object_bbox_in_camera(o), 0.9, "t") for o in world.objects if o.label == "screw"]
    mt = MultiTracker(backend="color")
    states = mt.start(dets, frame, color_hint="grey")
    assert len(states) == 2 and mt.active
    states = mt.update(cam.render())
    assert len(states) == 2
    mt.stop_all()
    assert not mt.active


def test_tracker_handles_tiny_bbox_without_crash():
    frame = np.zeros((100, 100, 3), np.uint8)
    tr = ObjectTracker()
    st = tr.start(Detection.from_bbox("x", BoundingBox(x=50, y=50, w=1, h=1), 0.5, "t"), frame)
    assert st is not None
    tr.update(frame)
