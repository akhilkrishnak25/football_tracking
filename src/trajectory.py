"""
trajectory.py
Maintains ball center coordinate history, calculates pixel movement metrics,
and renders visual trajectories (trail lines and markers) on video frames.
"""

import math
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np


class BallTrajectory:
    """
    Tracks and visualizes the movement path and trajectory of the sports ball.
    """

    def __init__(self, max_length: int = 100, enabled: bool = True):
        """
        Args:
            max_length: Maximum historical points to keep for rendering (0 or None for unlimited)
            enabled: Whether trajectory rendering is active
        """
        self.max_length = max_length
        self.enabled = enabled

        # Active trail buffer for drawing: deque of (x, y, track_id)
        # If max_length > 0, bounded deque; if <= 0, unbounded deque
        self.trail = deque(maxlen=max_length if max_length > 0 else None)

        # Full historical log across all frames for analytics
        self.history: List[Dict[str, Any]] = []

        # Previous point for instantaneous movement calculation
        self.last_point: Optional[Tuple[int, int]] = None

    def update_max_length(self, new_length: int):
        """Update trajectory trail length dynamically."""
        self.max_length = new_length
        current_items = list(self.trail)
        if new_length > 0:
            self.trail = deque(current_items[-new_length:], maxlen=new_length)
        else:
            self.trail = deque(current_items, maxlen=None)

    def add_point(
        self,
        frame_idx: int,
        center: Tuple[int, int],
        track_id: int,
        confidence: float
    ) -> float:
        """
        Record a new ball detection center.

        Args:
            frame_idx: Video frame index
            center: (cx, cy) pixel coordinates
            track_id: Persistent tracking ID
            confidence: Detection confidence score

        Returns:
            Displacement in pixels from the previous frame
        """
        cx, cy = center
        movement = 0.0

        if self.last_point is not None:
            px, py = self.last_point
            movement = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

        self.last_point = (cx, cy)
        self.trail.append((cx, cy, track_id))

        self.history.append({
            "frame": frame_idx,
            "x": cx,
            "y": cy,
            "track_id": track_id,
            "conf": confidence,
            "movement": round(movement, 2)
        })

        return movement

    def record_missed_frame(self, frame_idx: int):
        """Record a frame where no ball was detected."""
        self.last_point = None  # Reset instantaneous movement so gap isn't treated as a jump
        self.history.append({
            "frame": frame_idx,
            "x": None,
            "y": None,
            "track_id": None,
            "conf": 0.0,
            "movement": 0.0
        })

    def draw_trajectory(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the trajectory trail onto the frame with dynamic fading / gradient.

        Args:
            frame: BGR numpy image frame to annotate
        Returns:
            Annotated frame
        """
        if not self.enabled or len(self.trail) < 2:
            return frame

        points = list(self.trail)
        num_points = len(points)

        # Draw connecting line segments with fading thickness and color gradient
        for i in range(1, num_points):
            pt1 = (points[i - 1][0], points[i - 1][1])
            pt2 = (points[i][0], points[i][1])

            # Normalized progress through trail (0.0 to 1.0)
            progress = i / max(num_points, 1)

            # Color gradient: Cyan/Yellow (older) -> Neon Green / Red (newest)
            # BGR format: transition from (0, 165, 255) Orange to (0, 255, 0) Green
            b = int(0 * progress + 0 * (1 - progress))
            g = int(255 * progress + 165 * (1 - progress))
            r = int(50 * progress + 255 * (1 - progress))
            color = (b, g, r)

            thickness = max(1, int(1 + 3 * progress))
            cv2.line(frame, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

            # Draw small trail markers every 3 frames
            if i % 3 == 0 or i == num_points - 1:
                cv2.circle(frame, pt2, max(2, int(3 * progress)), color, -1, lineType=cv2.LINE_AA)

        # Draw pulsing outer ring at current ball center
        latest_pt = (points[-1][0], points[-1][1])
        cv2.circle(frame, latest_pt, 6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, latest_pt, 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        return frame

    def clear(self):
        """Reset trajectory buffer."""
        self.trail.clear()
        self.history.clear()
        self.last_point = None
