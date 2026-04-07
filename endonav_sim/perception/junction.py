"""Junction detector: find dark blobs (lumen openings) in a rendered frame.

Coaxial illumination on a ureteroscope means brightness falls as 1/r^2, so
distant openings show up as dark patches in the image. Counting and locating
these patches yields a coarse classification of the current view.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


@dataclass
class Blob:
    centroid_xy: tuple[int, int]
    area: int
    area_fraction: float
    mean_intensity: float
    contour: np.ndarray
    polar_image: tuple[float, float]  # (angle_rad, distance_norm) in image frame
    polar_world: tuple[float, float]  # angle de-rotated by cumulative roll


@dataclass
class JunctionResult:
    classification: Literal["junction", "lumen", "dead_end", "uncertain"]
    blobs: list[Blob]
    confirmed: bool
    dark_mask: np.ndarray


Classification = Literal["junction", "lumen", "dead_end", "uncertain"]


class JunctionDetector:
    """Adaptive dark-blob detector with temporal confirmation.

    The detector calibrates an intensity threshold from the running distribution
    of frame brightness over the last 50 frames; for the first 20 frames it
    returns ``classification='uncertain'`` while the buffer fills.
    """

    def __init__(
        self,
        min_blob_fraction: float = 0.01,
        confirm_frames_junction: int = 3,
        confirm_frames_deadend: int = 5,
    ) -> None:
        self.min_blob_fraction = float(min_blob_fraction)
        self.confirm_frames_junction = int(confirm_frames_junction)
        self.confirm_frames_deadend = int(confirm_frames_deadend)
        self._brightness_history: deque[float] = deque(maxlen=50)
        self._class_history: deque[Classification] = deque(
            maxlen=max(confirm_frames_junction, confirm_frames_deadend)
        )
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._aperture: np.ndarray | None = None  # cached circular ROI mask

    def process(self, frame: np.ndarray, cumulative_roll: float = 0.0) -> JunctionResult:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        blur = cv2.GaussianBlur(gray, (15, 15), 5)

        self._brightness_history.append(float(blur.mean()))

        H, W = blur.shape[:2]
        empty_mask = np.zeros_like(blur, dtype=np.uint8)

        if len(self._brightness_history) < 20:
            self._class_history.append("uncertain")
            return JunctionResult(
                classification="uncertain",
                blobs=[],
                confirmed=False,
                dark_mask=empty_mask,
            )

        # Threshold strategy: 40th percentile across the running history of
        # frame means gives a baseline expectation for the *frame* being dark;
        # the 40th percentile of *pixel* intensities in the current frame is
        # what actually separates lumen openings from mucosa. Take the smaller
        # of the two so a single bright frame (e.g. parked against a wall)
        # can't drag the threshold up.
        history_thresh = float(np.percentile(np.asarray(self._brightness_history), 40))
        frame_thresh = float(np.percentile(blur, 40))
        threshold = min(history_thresh, frame_thresh)
        mask = (blur < threshold).astype(np.uint8) * 255

        # Restrict to the circular endoscope aperture so the letterbox bars
        # and the heavily-vignetted edge can't masquerade as dark blobs.
        if self._aperture is None or self._aperture.shape != blur.shape:
            H_, W_ = blur.shape[:2]
            yy, xx = np.ogrid[:H_, :W_]
            cy_ = (H_ - 1) / 2.0
            cx_ = (W_ - 1) / 2.0
            radius = 0.92 * min(H_, W_) / 2.0
            self._aperture = (
                ((xx - cx_) ** 2 + (yy - cy_) ** 2) < radius**2
            ).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, self._aperture)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        total_area = float(H * W)
        min_area = self.min_blob_fraction * total_area
        cx_img = W / 2.0
        cy_img = H / 2.0
        radius_norm = min(H, W) / 2.0

        blobs: list[Blob] = []
        for label_id in range(1, n_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            cx, cy = centroids[label_id]
            dx = float(cx - cx_img)
            dy = float(cy - cy_img)
            angle_image = float(np.arctan2(-dy, dx))
            distance_norm = float(np.hypot(dx, dy) / radius_norm)
            polar_image = (angle_image, distance_norm)
            polar_world = (angle_image + float(cumulative_roll), distance_norm)

            comp_mask = (labels == label_id).astype(np.uint8)
            mean_intensity = float(blur[comp_mask.astype(bool)].mean())
            # A real lumen opening must be appreciably darker than the bright
            # mucosa that surrounds it. If the "dark" blob is only marginally
            # below the frame mean it is just dim wall, not an opening.
            frame_mean = float(blur.mean())
            if mean_intensity > 0.5 * frame_mean:
                continue
            contours, _ = cv2.findContours(
                comp_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour = contours[0] if contours else np.empty((0, 1, 2), dtype=np.int32)

            blobs.append(
                Blob(
                    centroid_xy=(int(round(cx)), int(round(cy))),
                    area=area,
                    area_fraction=area / total_area,
                    mean_intensity=mean_intensity,
                    contour=contour,
                    polar_image=polar_image,
                    polar_world=polar_world,
                )
            )

        # Sort largest first — useful for downstream consumers picking the
        # primary lumen.
        blobs.sort(key=lambda b: b.area, reverse=True)

        # If the dark mask saturates the frame (camera tip parked against the
        # back wall of a calyx), there is no real opening — call it a dead end
        # even though connected components technically yields one giant blob.
        dark_fraction = float(mask.sum() / 255.0 / total_area)
        if len(blobs) == 0 or dark_fraction > 0.55:
            classification: Classification = "dead_end"
            if dark_fraction > 0.55:
                blobs = []
        elif len(blobs) == 1:
            classification = "lumen"
        else:
            classification = "junction"

        self._class_history.append(classification)
        n_required = (
            self.confirm_frames_deadend
            if classification == "dead_end"
            else self.confirm_frames_junction
        )
        confirmed = (
            len(self._class_history) >= n_required
            and all(c == classification for c in list(self._class_history)[-n_required:])
        )

        return JunctionResult(
            classification=classification,
            blobs=blobs,
            confirmed=confirmed,
            dark_mask=mask,
        )
