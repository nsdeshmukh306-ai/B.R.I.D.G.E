"""A specialist, local, offline instrument recognizer.

BRIDGE's default source of *what* something is has been Gemini: capable at
general scene understanding, weak at telling two similar steel instruments
apart, and — being a network call — unavailable exactly where this project
now aims to work best (a rural hospital or a PHC with no reliable internet).

This module adds a second, narrower source of labels: a local object-detection
model trained specifically on surgical instruments, run entirely on-device
through ONNX Runtime. It does not replace Gemini — it is tried first, because
a specialist beats a generalist at this one job, and because it needs no
network at all. Gemini remains the fallback for anything the specialist
model doesn't recognise or when no specialist model is configured.

Nothing here changes what the rest of BRIDGE does with a label. Output still
goes through exactly the same path as a Gemini scene understanding result —
``SceneGraph.apply_ai()`` — so every invariant that already holds for AI
labels (local CV owns position, the model never emits anything but a label
and a box, no executable output) holds for this too. See CLAUDE.md.

Weights are never bundled with BRIDGE. See docs/recognition.md for what was
evaluated, why nothing is shipped by default, and the licensing constraints a
deployer needs to check before pointing this at a pretrained model.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("bridge.vision.recognizer")


@dataclass
class RecognizedInstrument:
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]  # pixel coords in the ORIGINAL frame


@dataclass
class RecognizerStatus:
    loaded: bool = False
    backend: str = "none"
    model_path: str = ""
    num_classes: int = 0
    last_error: str = ""
    last_run_ms: Optional[float] = None
    last_detection_count: Optional[int] = None
    history: list[dict] = field(default_factory=list)

    def record(self, latency_ms: float, count: int) -> None:
        self.last_run_ms = latency_ms
        self.last_detection_count = count
        self.history.append({"latency_ms": round(latency_ms, 2), "count": count})
        self.history = self.history[-50:]


def decode_yolo_onnx_output(
    output: np.ndarray,
    labels: list[str],
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.45,
) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    """Decode a raw Ultralytics-style YOLO ONNX output tensor.

    Accepts either the v8+ export shape (1, 4 + num_classes, num_boxes) or the
    v5 shape (1, num_boxes, 5 + num_classes). Boxes are centre-xywh in the
    model's own input pixel space (the caller rescales to the original frame).
    Pure function — no I/O, no model — so it is unit-testable without ONNX
    Runtime or a real weights file.

    Returns a list of (class_index, confidence, (x1, y1, x2, y2)), already
    filtered by confidence and passed through class-wise NMS.
    """
    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    nc = len(labels)

    if arr.shape[0] == 4 + nc:           # v8+/v11 export: (4+nc, N), no objectness
        arr = arr.T                       # -> (N, 4+nc)
        boxes_cxcywh = arr[:, :4]
        class_scores = arr[:, 4:4 + nc]
        obj = np.ones(arr.shape[0], dtype=np.float32)
    elif arr.shape[1] == 5 + nc:          # v5-style export: (N, 5+nc), with objectness
        boxes_cxcywh = arr[:, :4]
        obj = arr[:, 4]
        class_scores = arr[:, 5:5 + nc]
    else:
        raise ValueError(
            f"unexpected YOLO ONNX output shape {output.shape} for {nc} classes "
            "(expected (4+nc, N) or (N, 5+nc))"
        )

    class_idx = np.argmax(class_scores, axis=1)
    class_conf = class_scores[np.arange(len(class_scores)), class_idx]
    conf = obj * class_conf

    keep_mask = conf >= conf_threshold
    if not np.any(keep_mask):
        return []

    boxes_cxcywh = boxes_cxcywh[keep_mask]
    class_idx = class_idx[keep_mask]
    conf = conf[keep_mask]

    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    xywh_for_nms = [[float(x1[i]), float(y1[i]), float(w[i]), float(h[i])] for i in range(len(conf))]

    indices = cv2.dnn.NMSBoxes(xywh_for_nms, conf.tolist(), conf_threshold, iou_threshold)
    if indices is None or len(indices) == 0:
        return []
    indices = np.asarray(indices).flatten()

    out: list[tuple[int, float, tuple[float, float, float, float]]] = []
    for i in indices:
        out.append((int(class_idx[i]), float(conf[i]),
                    (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]))))
    return out


def _load_labels(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


class LocalInstrumentRecognizer:
    """Loads a user-supplied ONNX instrument-detection model and runs it locally.

    Disabled (``self.status.loaded`` stays False) until both a weights file
    and a labels file are found on disk and ONNX Runtime is importable. Every
    failure mode degrades to "the specialist model isn't available" rather
    than raising — this sits in the same optional position Gemini already
    occupies, so its absence must never break anything.
    """

    def __init__(self, weights_path: Path | str, labels_path: Path | str,
                 conf_threshold: float = 0.45, iou_threshold: float = 0.45,
                 input_size: int = 640):
        self.weights_path = Path(weights_path)
        self.labels_path = Path(labels_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.status = RecognizerStatus(model_path=str(self.weights_path))
        self._session = None
        self._input_name: str = ""
        self.labels: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self.weights_path.exists():
            self.status.last_error = f"no weights file at {self.weights_path}"
            return
        if not self.labels_path.exists():
            self.status.last_error = f"no labels file at {self.labels_path}"
            return
        try:
            import onnxruntime as ort  # optional dependency — see pyproject.toml [recognizer]
        except ImportError:
            self.status.last_error = "onnxruntime is not installed (pip install bridge-spatial[recognizer])"
            return
        try:
            self.labels = _load_labels(self.labels_path)
            self._session = ort.InferenceSession(str(self.weights_path), providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            self.status.loaded = True
            self.status.backend = "onnxruntime"
            self.status.num_classes = len(self.labels)
            log.info("Local instrument recognizer loaded: %d classes from %s", len(self.labels), self.weights_path)
        except Exception as e:  # noqa: BLE001 - a bad/incompatible model file must not crash BRIDGE
            self.status.last_error = f"failed to load model: {e}"
            self._session = None

    @property
    def available(self) -> bool:
        return self.status.loaded and self._session is not None

    def recognize(self, frame: np.ndarray) -> list[RecognizedInstrument]:
        """Run detection on one frame. Returns [] if the model isn't loaded."""
        if not self.available:
            return []
        t0 = time.perf_counter()
        try:
            h, w = frame.shape[:2]
            scale = self.input_size / max(h, w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            resized = cv2.resize(frame, (nw, nh))
            canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            canvas[:nh, :nw] = resized
            blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, 0..1
            blob = np.transpose(blob, (2, 0, 1))[None, ...]       # NCHW

            outputs = self._session.run(None, {self._input_name: blob})
            decoded = decode_yolo_onnx_output(outputs[0], self.labels, self.conf_threshold, self.iou_threshold)

            results = []
            for class_idx, conf, (x1, y1, x2, y2) in decoded:
                # undo letterbox scaling back to the original frame
                results.append(RecognizedInstrument(
                    label=self.labels[class_idx],
                    confidence=conf,
                    box_xyxy=(x1 / scale, y1 / scale, x2 / scale, y2 / scale),
                ))
            self.status.record((time.perf_counter() - t0) * 1000, len(results))
            return results
        except Exception as e:  # noqa: BLE001 - perception must never take the app down
            self.status.last_error = f"inference failed: {e}"
            log.exception("local instrument recognizer inference failed")
            return []


def build_recognizer(weights_path, labels_path, conf_threshold: float = 0.45,
                      iou_threshold: float = 0.45, input_size: int = 640) -> Optional[LocalInstrumentRecognizer]:
    """Construct a recognizer, or return None if no weights/labels are configured.

    Kept separate from the class so callers (and settings wiring) can pass
    ``None``/empty paths and get a clean "not configured" rather than a
    FileNotFoundError-shaped code path to guard against everywhere.
    """
    if not weights_path or not labels_path:
        return None
    return LocalInstrumentRecognizer(weights_path, labels_path, conf_threshold, iou_threshold, input_size)
