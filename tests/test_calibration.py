import numpy as np

from bridge.spatial.calibration import PlanarMarkerCalibration, marker_layout
from bridge.spatial.geometry import Point
from bridge.spatial.homography import apply_homography, compute_homography, points_to_array
from bridge.spatial.markers import detect_bright_markers, match_markers_to_layout
from bridge.spatial.profile import CalibrationProfile, ProfileStore
from bridge.spatial.validation import CalibrationValidator


def test_validator_known_transform_gives_expected_error():
    src = points_to_array([Point(x=0, y=0), Point(x=100, y=0), Point(x=100, y=100), Point(x=0, y=100)])
    dst = src * 2
    H, _ = compute_homography(src, dst)
    # perturb targets by exactly 3 px in x
    res = CalibrationValidator(threshold_px=5).validate(H, src, dst + np.array([3.0, 0.0]))
    assert abs(res.mean_error_px - 3.0) < 1e-6 and abs(res.max_error_px - 3.0) < 1e-6
    assert res.valid
    res_bad = CalibrationValidator(threshold_px=2).validate(H, src, dst + np.array([3.0, 0.0]))
    assert not res_bad.valid


def test_validator_rejects_too_few_points():
    res = CalibrationValidator().validate(np.eye(3), np.zeros((2, 2)), np.zeros((2, 2)))
    assert not res.valid and res.n_points == 2


def test_marker_detection_on_synthetic_frame():
    import cv2

    ref = np.full((300, 400, 3), 40, np.uint8)
    frame = ref.copy()
    pts = [(50, 60), (350, 70), (340, 240), (60, 230)]
    for p in pts:
        cv2.circle(frame, p, 12, (255, 255, 255), -1, cv2.LINE_AA)
    dets = detect_bright_markers(frame, ref, expected_count=4)
    assert len(dets) == 4
    for d in dets:
        assert min(np.hypot(d.center.x - x, d.center.y - y) for x, y in pts) < 1.0
    matched = match_markers_to_layout(dets, [Point(x=x, y=y) for x, y in pts], (400, 300))
    assert [i for i, _ in matched] == [0, 1, 2, 3]


def test_marker_layout_shapes():
    assert len(marker_layout(1280, 720, 4)) == 4
    assert len(marker_layout(1280, 720, 9)) == 9


def test_planar_calibration_matches_simulator_ground_truth(sim, cam_link):
    world, proj, cam = sim
    res = PlanarMarkerCalibration(settle_s=0).calibrate(proj, cam_link)
    assert res.success and res.validation is not None and res.validation.valid
    assert res.validation.mean_error_px < 5.0
    H = res.homography.to_numpy()
    G = cam.ground_truth_cam_to_proj()
    pts = np.array([[200, 150], [500, 300], [800, 450], [100, 500]], float)
    err = np.linalg.norm(apply_homography(H, pts) - apply_homography(G, pts), axis=1)
    assert err.max() < 3.0


def test_nine_point_calibration_also_valid(sim, cam_link):
    world, proj, cam = sim
    res = PlanarMarkerCalibration(n_points=9, settle_s=0).calibrate(proj, cam_link)
    assert res.success and res.validation.mean_error_px < 5.0


def test_calibration_fails_gracefully_when_camera_sees_nothing(sim):
    world, proj, cam = sim

    class BlindCamera:
        def capture(self, settle_s=0.0):
            return np.zeros((540, 960, 3), np.uint8)

        def size(self):
            return (960, 540)

    res = PlanarMarkerCalibration(settle_s=0, max_retries=1).calibrate(proj, BlindCamera())
    assert not res.success
    assert "Possible causes" in res.message


def test_profile_store_round_trip_and_hardware_matching(tmp_path, sim, cam_link):
    world, proj, cam = sim
    method = PlanarMarkerCalibration(settle_s=0)
    res = method.calibrate(proj, cam_link)
    store = ProfileStore(tmp_path)
    prof = method.save(res, store, name="desk setup", camera_id="cam:0", camera_resolution=(960, 540),
                       display_id="display:1", display_resolution=(1280, 720), surface_type="table")
    loaded = store.load("desk setup")
    assert loaded is not None and np.allclose(loaded.homography.to_numpy(), prof.homography.to_numpy())
    assert store.find_matching("cam:0", (960, 540), "display:1", (1280, 720)) is not None
    assert store.find_matching("cam:1", (960, 540), "display:1", (1280, 720)) is None
    ok, reason = loaded.matches_hardware("cam:0", (1920, 1080), "display:1", (1280, 720))
    assert not ok and "resolution" in reason.lower()
    ok, reason = loaded.matches_hardware("cam:0", (960, 540), "display:2", (1280, 720))
    assert not ok and "display" in reason.lower()
    assert store.delete("desk setup") and store.load("desk setup") is None


def test_corrupt_profile_is_skipped(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    assert ProfileStore(tmp_path).list() == []
