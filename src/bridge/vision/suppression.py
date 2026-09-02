"""Self-projection suppression.

A projector-camera loop sees its own graphics. Given the projector-space mask
of what BRIDGE is currently drawing and the camera->projector homography, we
warp the mask into camera space and replace those pixels with a local
background estimate, so detection/tracking never latches onto projected light.
"""
from __future__ import annotations

import cv2
import numpy as np


def projector_mask_to_camera(mask_proj: np.ndarray, H_cam_to_proj: np.ndarray, cam_size: tuple[int, int],
                             dilate_px: int = 6) -> np.ndarray:
    w, h = cam_size
    H_proj_to_cam = np.linalg.inv(H_cam_to_proj)
    m = cv2.warpPerspective(mask_proj, H_proj_to_cam, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
    return m


def suppress(frame: np.ndarray, mask_cam: np.ndarray) -> np.ndarray:
    """Replace masked pixels with a heavily blurred (background-like) version of the frame."""
    if mask_cam is None or not mask_cam.any():
        return frame
    small = cv2.resize(frame, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 7)
    bg = cv2.resize(small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = frame.copy()
    sel = mask_cam > 0
    out[sel] = bg[sel]
    return out
