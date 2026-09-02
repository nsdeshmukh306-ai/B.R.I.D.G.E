"""Hardware-free simulation: a virtual table, virtual objects, a virtual projector
whose light falls on the table, and a virtual camera that photographs it.

The virtual camera is a real FrameSource, and the virtual projector is a real
ProjectorLink, so the *same* calibration, tracking and rendering code runs in
simulation and with physical hardware. Ground-truth homographies are known, so
calibration accuracy can be asserted in tests.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import cv2
import numpy as np

from bridge.camera.device import FrameSource
from bridge.spatial.geometry import BoundingBox, Point
from bridge.spatial.homography import apply_homography

Shape = Literal["rect", "ellipse", "screwdriver", "screw", "pen", "scissors", "phone"]


@dataclass
class VirtualObject:
    id: str
    label: str
    x: float  # table coordinates (px in table image)
    y: float
    w: float
    h: float
    color: tuple[int, int, int] = (60, 60, 200)  # BGR
    shape: Shape = "rect"
    angle_deg: float = 0.0
    description: str = ""

    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox(x=self.x - self.w / 2, y=self.y - self.h / 2, w=self.w, h=self.h)

    @property
    def center(self) -> Point:
        return Point(x=self.x, y=self.y)


def default_objects() -> list[VirtualObject]:
    return [
        VirtualObject("obj-screwdriver", "screwdriver", 380, 300, 220, 40, (40, 40, 210), "screwdriver", 20,
                      "a red-handled screwdriver"),
        VirtualObject("obj-phone", "phone", 720, 200, 90, 170, (30, 30, 30), "phone", 0, "a black smartphone"),
        VirtualObject("obj-scissors", "scissors", 700, 450, 160, 70, (160, 60, 20), "scissors", -30, "blue-handled scissors"),
        VirtualObject("obj-screw-1", "screw", 220, 480, 22, 22, (90, 90, 90), "screw", 0, "a small grey screw"),
        VirtualObject("obj-screw-2", "screw", 265, 500, 22, 22, (90, 90, 90), "screw", 0, "a small grey screw"),
        VirtualObject("obj-pen", "pen", 500, 520, 170, 18, (200, 120, 20), "pen", 5, "a blue pen"),
    ]


class VirtualWorld:
    """Table image space is the 'physical' coordinate system of the simulation."""

    def __init__(self, table_w: int = 1000, table_h: int = 620, objects: list[VirtualObject] | None = None,
                 table_color: tuple[int, int, int] = (235, 240, 245)):
        self.table_w, self.table_h = table_w, table_h
        self.table_color = table_color
        self.objects: list[VirtualObject] = objects if objects is not None else default_objects()
        self._lock = threading.RLock()

    def get(self, obj_id: str) -> Optional[VirtualObject]:
        return next((o for o in self.objects if o.id == obj_id), None)

    def move(self, obj_id: str, x: float, y: float) -> None:
        with self._lock:
            o = self.get(obj_id)
            if o:
                o.x, o.y = float(np.clip(x, 0, self.table_w)), float(np.clip(y, 0, self.table_h))

    def move_by(self, obj_id: str, dx: float, dy: float) -> None:
        o = self.get(obj_id)
        if o:
            self.move(obj_id, o.x + dx, o.y + dy)

    def remove(self, obj_id: str) -> None:
        with self._lock:
            self.objects = [o for o in self.objects if o.id != obj_id]

    def object_at(self, p: Point) -> Optional[VirtualObject]:
        for o in reversed(self.objects):
            b = o.bbox
            if b.x <= p.x <= b.x2 and b.y <= p.y <= b.y2:
                return o
        return None

    def render_table(self) -> np.ndarray:
        img = np.empty((self.table_h, self.table_w, 3), np.uint8)
        img[:] = self.table_color
        # subtle wood-grain-ish texture so trackers have something to hold on to
        yy = np.arange(self.table_h)[:, None]
        grain = ((np.sin(yy / 9.0) * 4).astype(np.int16))
        img = np.clip(img.astype(np.int16) + grain[:, :, None], 0, 255).astype(np.uint8)
        with self._lock:
            for o in self.objects:
                self._draw_object(img, o)
        return img

    @staticmethod
    def _rot_rect(cx, cy, w, h, ang) -> np.ndarray:
        return cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.int32)

    def _draw_object(self, img: np.ndarray, o: VirtualObject) -> None:
        c = o.color
        dark = tuple(int(v * 0.6) for v in c)
        if o.shape == "screwdriver":
            handle = self._rot_rect(o.x - o.w * 0.25, o.y, o.w * 0.5, o.h, o.angle_deg)
            shaft = self._rot_rect(o.x + o.w * 0.25, o.y, o.w * 0.5, o.h * 0.3, o.angle_deg)
            cv2.fillPoly(img, [shaft], (170, 170, 175), cv2.LINE_AA)
            cv2.fillPoly(img, [handle], c, cv2.LINE_AA)
            cv2.polylines(img, [handle], True, dark, 2, cv2.LINE_AA)
        elif o.shape == "screw":
            cv2.circle(img, (int(o.x), int(o.y)), int(o.w / 2), c, -1, cv2.LINE_AA)
            cv2.line(img, (int(o.x - o.w / 3), int(o.y)), (int(o.x + o.w / 3), int(o.y)), (40, 40, 40), 2, cv2.LINE_AA)
        elif o.shape == "ellipse":
            cv2.ellipse(img, (int(o.x), int(o.y)), (int(o.w / 2), int(o.h / 2)), o.angle_deg, 0, 360, c, -1, cv2.LINE_AA)
        elif o.shape == "scissors":
            b1 = self._rot_rect(o.x + o.w * 0.25, o.y - o.h * 0.15, o.w * 0.5, o.h * 0.18, o.angle_deg)
            b2 = self._rot_rect(o.x + o.w * 0.25, o.y + o.h * 0.15, o.w * 0.5, o.h * 0.18, o.angle_deg)
            cv2.fillPoly(img, [b1], (180, 180, 185), cv2.LINE_AA)
            cv2.fillPoly(img, [b2], (180, 180, 185), cv2.LINE_AA)
            for dy in (-0.25, 0.25):
                cv2.ellipse(img, (int(o.x - o.w * 0.3), int(o.y + o.h * dy)), (int(o.w * 0.18), int(o.h * 0.22)),
                            o.angle_deg, 0, 360, c, 4, cv2.LINE_AA)
        elif o.shape == "phone":
            r = self._rot_rect(o.x, o.y, o.w, o.h, o.angle_deg)
            cv2.fillPoly(img, [r], c, cv2.LINE_AA)
            inner = self._rot_rect(o.x, o.y, o.w * 0.85, o.h * 0.9, o.angle_deg)
            cv2.fillPoly(img, [inner], (70, 60, 50), cv2.LINE_AA)
        else:  # rect / pen
            r = self._rot_rect(o.x, o.y, o.w, o.h, o.angle_deg)
            cv2.fillPoly(img, [r], c, cv2.LINE_AA)
            cv2.polylines(img, [r], True, dark, 1, cv2.LINE_AA)


def _quad_homography(src_quad: np.ndarray, dst_quad: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(src_quad.astype(np.float32), dst_quad.astype(np.float32))


class SimulatedProjector:
    """Implements ProjectorLink. Its image lands on the table through H_proj_to_table."""

    def __init__(self, width: int = 1280, height: int = 720, table_quad: np.ndarray | None = None,
                 world: VirtualWorld | None = None, brightness: float = 0.85):
        self.width, self.height = width, height
        self.world = world
        tw, th = (world.table_w, world.table_h) if world else (1000, 620)
        # Where the projector's four corners land on the table (slightly keystoned by default).
        if table_quad is None:
            table_quad = np.array([[90, 60], [tw - 60, 40], [tw - 40, th - 50], [60, th - 70]], np.float32)
        self.table_quad = table_quad
        proj_quad = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
        self.H_proj_to_table = _quad_homography(proj_quad, table_quad)
        self.brightness = brightness
        self._image = np.full((height, width, 3), 255, np.uint8)
        self._lock = threading.Lock()
        self.version = 0
        self._light_cache: tuple[int, Optional[np.ndarray]] = (-1, None)

    def show_image(self, image: np.ndarray) -> None:
        with self._lock:
            if image.shape[0] != self.height or image.shape[1] != self.width:
                image = cv2.resize(image, (self.width, self.height))
            self._image = image.copy()
            self.version += 1

    def current_image(self) -> np.ndarray:
        with self._lock:
            return self._image.copy()

    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def light_on_table(self, tw: int, th: int) -> np.ndarray:
        """Projected light map in table space (float 0..1 per channel). Cached per shown image."""
        with self._lock:
            version, cached = self._light_cache
            if cached is not None and version == self.version and cached.shape[:2] == (th, tw):
                return cached
            img = self._image
            v = self.version
        warped = cv2.warpPerspective(img, self.H_proj_to_table, (tw, th), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        light = warped.astype(np.float32) / 255.0
        with self._lock:
            self._light_cache = (v, light)
        return light


class SimulatedCamera(FrameSource):
    """A FrameSource that photographs the lit virtual table with perspective + noise."""

    def __init__(self, world: VirtualWorld, projector: SimulatedProjector | None, width: int = 960, height: int = 540,
                 camera_quad: np.ndarray | None = None, noise_sigma: float = 2.0, ambient: float = 0.45,
                 fps: float = 30.0, seed: int = 0):
        self.world, self.projector = world, projector
        self.width, self.height = width, height
        tw, th = world.table_w, world.table_h
        # Where the table's corners appear in the camera image (a mild perspective view).
        if camera_quad is None:
            camera_quad = np.array([[70, 40], [width - 40, 55], [width - 15, height - 25], [30, height - 45]], np.float32)
        table_quad = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
        self.H_table_to_cam = _quad_homography(table_quad, camera_quad)
        self.H_cam_to_table = np.linalg.inv(self.H_table_to_cam)
        self.noise_sigma, self.ambient, self.fps = noise_sigma, ambient, fps
        self._rng = np.random.default_rng(seed)
        # Pre-generated sensor-noise bank: per-frame Gaussian sampling at full resolution is slow.
        self._noise = [self._rng.normal(0, 1.0, (height, width, 3)).astype(np.float32) for _ in range(4)]
        self._last = 0.0
        self._open = True
        self.frames = 0

    # ground truth for tests
    def ground_truth_cam_to_proj(self) -> np.ndarray:
        assert self.projector is not None
        H = np.linalg.inv(self.projector.H_proj_to_table) @ self.H_cam_to_table
        return H / H[2, 2]

    def table_to_camera(self, p: Point) -> Point:
        out = apply_homography(self.H_table_to_cam, [p])[0]
        return Point(x=float(out[0]), y=float(out[1]))

    def camera_to_table(self, p: Point) -> Point:
        out = apply_homography(self.H_cam_to_table, [p])[0]
        return Point(x=float(out[0]), y=float(out[1]))

    def object_bbox_in_camera(self, o: VirtualObject) -> BoundingBox:
        pts = apply_homography(self.H_table_to_cam, o.bbox.corners())
        return BoundingBox.from_xyxy(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def is_open(self) -> bool:
        return self._open

    def release(self) -> None:
        self._open = False

    def render(self) -> np.ndarray:
        table = self.world.render_table().astype(np.float32) / 255.0
        if self.projector is not None:
            light = self.projector.light_on_table(self.world.table_w, self.world.table_h)
            lit = table * (self.ambient + (1 - self.ambient) * self.projector.brightness * light + 0.05)
        else:
            lit = table
        lit = np.clip(lit, 0, 1)
        frame = cv2.warpPerspective(lit, self.H_table_to_cam, (self.width, self.height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0.12, 0.12, 0.13))
        frame = frame * 255.0
        if self.noise_sigma > 0:
            frame += self._noise[self.frames % len(self._noise)] * self.noise_sigma
        self.frames += 1
        return np.clip(frame, 0, 255).astype(np.uint8)

    def read_frame(self) -> Optional[np.ndarray]:
        if not self._open:
            return None
        # Pace to the configured fps so the capture thread does not spin.
        now = time.perf_counter()
        wait = (1.0 / self.fps) - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.perf_counter()
        return self.render()


def make_default_simulation(seed: int = 0) -> tuple[VirtualWorld, SimulatedProjector, SimulatedCamera]:
    world = VirtualWorld()
    projector = SimulatedProjector(world=world)
    camera = SimulatedCamera(world, projector, seed=seed)
    return world, projector, camera
