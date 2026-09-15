"""The local, offline instrument recognizer: the pure ONNX-output decoder,
graceful degradation when no model is configured, and its place ahead of
Gemini in the scene-labelling pipeline."""
import time

import numpy as np
import pytest

from bridge.spatial.geometry import BoundingBox
from bridge.vision.surgical_recognizer import (
    LocalInstrumentRecognizer,
    RecognizedInstrument,
    build_recognizer,
    decode_yolo_onnx_output,
)


# --- decode_yolo_onnx_output: pure function, no model or ONNX Runtime needed --------------
LABELS = ["scalpel", "artery clamp", "mayo scissors"]


def _v8_output(cx, cy, w, h, class_idx, class_conf, nc=3):
    """Build a synthetic (1, 4+nc, N) v8-style tensor for one box."""
    row = np.zeros(4 + nc, dtype=np.float32)
    row[:4] = [cx, cy, w, h]
    row[4 + class_idx] = class_conf
    return row.reshape(4 + nc, 1)[None, ...]


def test_v8_shape_decodes_a_confident_box():
    out = _v8_output(cx=100, cy=100, w=40, h=40, class_idx=1, class_conf=0.9)
    decoded = decode_yolo_onnx_output(out, LABELS, conf_threshold=0.4)
    assert len(decoded) == 1
    class_idx, conf, (x1, y1, x2, y2) = decoded[0]
    assert LABELS[class_idx] == "artery clamp"
    assert conf == pytest.approx(0.9, abs=1e-4)
    assert (x1, y1, x2, y2) == pytest.approx((80, 80, 120, 120))


def test_v5_shape_decodes_with_objectness():
    """(1, N, 5+nc): objectness * class score is the confidence."""
    nc = 3
    row = np.zeros(5 + nc, dtype=np.float32)
    row[:4] = [50, 50, 20, 20]
    row[4] = 0.8          # objectness
    row[4 + 1 + 0] = 0.9  # class 0 (scalpel) score
    out = row.reshape(1, 5 + nc)
    decoded = decode_yolo_onnx_output(out, LABELS, conf_threshold=0.5)
    assert len(decoded) == 1
    class_idx, conf, _ = decoded[0]
    assert LABELS[class_idx] == "scalpel"
    assert conf == pytest.approx(0.72, abs=1e-4)   # 0.8 * 0.9


def test_low_confidence_boxes_are_dropped():
    out = _v8_output(cx=100, cy=100, w=40, h=40, class_idx=0, class_conf=0.1)
    assert decode_yolo_onnx_output(out, LABELS, conf_threshold=0.4) == []


def test_overlapping_boxes_are_reduced_by_nms():
    nc = 3
    a = np.zeros(4 + nc, dtype=np.float32)
    a[:4], a[4] = [100, 100, 40, 40], 0.9
    b = np.zeros(4 + nc, dtype=np.float32)
    b[:4], b[4] = [102, 101, 40, 40], 0.85   # near-identical box, same class
    out = np.stack([a, b], axis=1)[None, ...]  # (1, 4+nc, 2)
    decoded = decode_yolo_onnx_output(out, LABELS, conf_threshold=0.4, iou_threshold=0.5)
    assert len(decoded) == 1
    assert decoded[0][1] == pytest.approx(0.9, abs=1e-4)   # the more confident one survives


def test_wrong_shape_raises_a_clear_error():
    bad = np.zeros((1, 6, 5))  # neither 4+nc=7 nor 5+nc=8 for 3 classes
    with pytest.raises(ValueError, match="unexpected YOLO ONNX output shape"):
        decode_yolo_onnx_output(bad, LABELS)


# --- graceful degradation: no weights configured, or files missing ------------------------
def test_build_recognizer_returns_none_without_paths():
    assert build_recognizer(None, None) is None
    assert build_recognizer("", "") is None


def test_recognizer_degrades_cleanly_when_files_are_missing(tmp_path):
    r = LocalInstrumentRecognizer(tmp_path / "nope.onnx", tmp_path / "nope.txt")
    assert not r.available
    assert "no weights file" in r.status.last_error
    assert r.recognize(np.zeros((100, 100, 3), dtype=np.uint8)) == []


def test_recognizer_degrades_cleanly_when_onnxruntime_is_absent(tmp_path, monkeypatch):
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"not a real model")
    labels = tmp_path / "labels.txt"
    labels.write_text("scalpel\nartery clamp\n")

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "onnxruntime":
            raise ImportError("no onnxruntime in this environment")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    r = LocalInstrumentRecognizer(weights, labels)
    assert not r.available
    assert "onnxruntime is not installed" in r.status.last_error


def test_recognizer_reports_a_bad_model_file_without_raising(tmp_path):
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"this is not a valid onnx model")
    labels = tmp_path / "labels.txt"
    labels.write_text("scalpel\n")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        pytest.skip("onnxruntime not installed in this environment")
    r = LocalInstrumentRecognizer(weights, labels)
    assert not r.available
    assert r.status.last_error


# --- integration: local recognizer runs ahead of Gemini in JarvisAssistant ----------------
@pytest.fixture
def core(tmp_path):
    from bridge.app.application import BridgeCore
    from bridge.app.config import ConfigManager

    cfg = ConfigManager(tmp_path / "settings.yaml")
    cfg.settings.profiles_dir = tmp_path / "profiles"
    cfg.settings.log_dir = tmp_path / "logs"
    cfg.settings.assistant.records_dir = tmp_path / "records"
    cfg.settings.assistant.ai_label_interval_s = 1e6      # no background AI during tests
    c = BridgeCore(cfg)
    c.enter_simulation(seed=3, scene="workshop")
    c.calibration.method.settle_s = 0.02
    c.cameras.wait_for_frame(3.0)
    yield c
    c.shutdown()


class _StubRecognizer:
    """Duck-types LocalInstrumentRecognizer without needing a real model file."""

    def __init__(self, results):
        self._results = results
        self.status = type("S", (), {"loaded": True, "last_error": ""})()
        self.conf_threshold = 0.4

    @property
    def available(self):
        return True

    def recognize(self, frame):
        return self._results


def test_local_recognizer_labels_are_tagged_and_gemini_still_fills_gaps(core):
    """The specialist model runs first (source=specialist-local); Gemini still
    runs afterward (source=gemini) so anything outside the local model's
    trained classes still gets named — exactly the fallback behaviour
    described in docs/recognition.md."""
    a = core.assistant
    assert core.calibrate().success
    frame = core.cameras.wait_for_frame(2.0)

    a.recognizer = _StubRecognizer([
        RecognizedInstrument(label="artery clamp", confidence=0.93, box_xyxy=(80, 80, 120, 120)),
    ])
    calls = {"n": 0}
    original_understand = core.ai.understand_scene

    def counting_understand(frame):
        calls["n"] += 1
        return original_understand(frame)

    core.ai.understand_scene = counting_understand
    a._label_scene_async(frame)
    for _ in range(50):
        if not a._ai_labelling.is_set():
            break
        time.sleep(0.02)
    assert not a._ai_labelling.is_set(), "background labelling did not finish"

    ent = a.scene.find_one("artery clamp")
    assert ent is not None
    assert ent.source == "specialist-local"
    assert calls["n"] == 1, "Gemini must still be asked so it can label what the specialist model didn't"


def test_no_recognizer_configured_behaves_exactly_as_before(core):
    """With recognizer left at None (the default), only Gemini labels the scene —
    unchanged v0.4 behaviour."""
    a = core.assistant
    assert a.recognizer is None
    assert core.calibrate().success
    frame = core.cameras.wait_for_frame(2.0)
    a._label_scene_async(frame)
    for _ in range(50):
        if not a._ai_labelling.is_set():
            break
        time.sleep(0.02)
    assert not a._ai_labelling.is_set()
