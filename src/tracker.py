"""
tracker.py
Implements persistent object tracking for sports balls using ByteTrack and BoT-SORT.
Ensures tracking ID persistence across consecutive video frames and handles missed detections.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO


class BallTracker:
    """
    Manages multi-frame ball tracking using Ultralytics ByteTrack / BoT-SORT algorithms.
    """

    SUPPORTED_TRACKERS = {
        "ByteTrack": "bytetrack.yaml",
        "BoT-SORT": "botsort.yaml"
    }

    def __init__(
        self,
        model: YOLO,
        tracker_type: str = "ByteTrack",
        ball_class_ids: Optional[List[int]] = None,
        device: str = "cpu"
    ):
        """
        Initialize the ball tracker.

        Args:
            model: Instantiated YOLO model object
            tracker_type: 'ByteTrack' or 'BoT-SORT'
            ball_class_ids: Filtered class IDs for sports ball
            device: 'cuda:0' or 'cpu'
        """
        self.model = model
        self.tracker_type = tracker_type
        self.tracker_config = self.SUPPORTED_TRACKERS.get(tracker_type, "bytetrack.yaml")
        self.ball_class_ids = ball_class_ids or [32]
        self.device = device

        # Internal tracking state to maintain consistency if tracker ID temporarily drops
        self.last_active_id: Optional[int] = None
        self.fallback_id_counter: int = 1
        self.missed_frame_count: int = 0
        self.max_missed_frames: int = 30  # Reset fallback ID after 30 lost frames

    def reset(self):
        """Reset internal tracking state between video processing runs."""
        self.last_active_id = None
        self.fallback_id_counter = 1
        self.missed_frame_count = 0

    def track_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.50,
        imgsz: int = 640
    ) -> List[Dict[str, Any]]:
        """
        Process a single frame with persistent tracking.

        Args:
            frame: BGR numpy image frame
            conf_threshold: Minimum detection confidence (0.1 - 1.0)
            iou_threshold: IoU threshold for NMS and association
            imgsz: Inference image size

        Returns:
            List of track dicts: [
                {
                    "track_id": int,
                    "bbox": [x1, y1, x2, y2],
                    "conf": float,
                    "class_id": int,
                    "class_name": str,
                    "center": (cx, cy)
                }, ...
            ]
        """
        # Run persistent tracking inference
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=self.device,
            classes=self.ball_class_ids,
            verbose=False
        )

        tracks = []
        if not results:
            self._handle_missed_frame()
            return tracks

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self._handle_missed_frame()
            return tracks

        names = self.model.names
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = [int(coord) for coord in xyxy]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Retrieve tracking ID from tracker output
            if box.id is not None:
                track_id = int(box.id[0].item())
                self.last_active_id = track_id
                self.missed_frame_count = 0
            else:
                # If tracker hasn't assigned an ID yet, maintain last known ID or fallback
                if self.last_active_id is not None and self.missed_frame_count < 10:
                    track_id = self.last_active_id
                else:
                    track_id = self.fallback_id_counter
                    self.last_active_id = track_id

            class_name = names.get(cls_id, "sports ball") if isinstance(names, dict) else str(cls_id)

            tracks.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "conf": round(conf, 4),
                "class_id": cls_id,
                "class_name": class_name,
                "center": (cx, cy)
            })

        # Sort tracks by confidence descending
        tracks.sort(key=lambda t: t["conf"], reverse=True)
        return tracks

    def _handle_missed_frame(self):
        """Update counter when no ball is detected in the current frame."""
        self.missed_frame_count += 1
        if self.missed_frame_count > self.max_missed_frames:
            # If ball lost for too long, reset active track
            self.last_active_id = None
