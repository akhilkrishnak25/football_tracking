"""
detector.py
Handles YOLO11 model loading, hardware acceleration detection (CUDA / CPU),
and ball detection inference with confidence, IoU, and class filtering.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from ultralytics import YOLO


def get_device() -> Tuple[str, str]:
    """
    Detect available computing hardware.
    Returns:
        (device_str, display_str): e.g. ("cuda:0", "Device: CUDA (NVIDIA GeForce RTX ...)") or ("cpu", "Device: CPU")
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return "cuda:0", f"Device: CUDA ({gpu_name})"
    return "cpu", "Device: CPU"


class BallDetector:
    """
    Wrapper for YOLO11 ball detection.
    Supports YOLO11n, YOLO11s, YOLO11m, and custom-trained ball models.
    """

    COCO_SPORTS_BALL_CLASS_NAME = "sports ball"

    def __init__(
        self,
        model_name_or_path: str = "yolo11n.pt",
        use_custom_model: bool = False,
        custom_model_path: str = "models/best.pt",
        device: Optional[str] = None
    ):
        """
        Initialize detector with specified model.
        Args:
            model_name_or_path: Name of standard YOLO11 model or path
            use_custom_model: If True, loads custom_model_path
            custom_model_path: Filepath to custom trained YOLO11 weights
            device: 'cuda:0', 'cpu', or None for auto-detection
        """
        if device is None:
            self.device, self.device_display = get_device()
        else:
            self.device = device
            self.device_display = f"Device: {device.upper()}"

        self.use_custom = use_custom_model
        self.custom_model_path = custom_model_path

        if self.use_custom:
            if not os.path.exists(custom_model_path):
                raise FileNotFoundError(
                    f"Custom model file '{custom_model_path}' not found! "
                    f"Please place your trained weights in '{custom_model_path}' or train one using scripts/train.py."
                )
            self.model_path = custom_model_path
        else:
            self.model_path = model_name_or_path

        # Load YOLO model
        self.model = YOLO(self.model_path)
        self.ball_class_ids = self._find_ball_class_ids()

    def _find_ball_class_ids(self) -> List[int]:
        """
        Dynamically locate ball-related class indices from model.names.
        In COCO dataset, 'sports ball' is class ID 32.
        In custom datasets, 'ball' is typically class ID 0.
        """
        ball_ids = []
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            for class_id, name in names.items():
                name_lower = str(name).lower()
                if "ball" in name_lower:
                    ball_ids.append(int(class_id))
        elif isinstance(names, list):
            for class_id, name in enumerate(names):
                if "ball" in str(name).lower():
                    ball_ids.append(class_id)

        # If custom model and no 'ball' found by name, default to all classes in custom model
        if not ball_ids and self.use_custom:
            if isinstance(names, dict):
                ball_ids = [int(k) for k in names.keys()]
            elif isinstance(names, list):
                ball_ids = list(range(len(names)))
            else:
                ball_ids = [0]

        # Fallback to COCO class 32 if still empty on standard model
        if not ball_ids:
            ball_ids = [32]

        return ball_ids

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.50,
        imgsz: int = 640
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a single video frame and filter for ball detections.

        Args:
            frame: BGR numpy image frame
            conf_threshold: Minimum detection confidence (0.1 - 1.0)
            iou_threshold: NMS IoU threshold (0.1 - 1.0)
            imgsz: Inference image size

        Returns:
            List of dicts: [
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": float,
                    "class_id": int,
                    "class_name": str,
                    "center": (int_x, int_y)
                }, ...
            ]
        """
        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=self.device,
            classes=self.ball_class_ids,
            verbose=False
        )

        detections = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names = self.model.names
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = [int(coord) for coord in xyxy]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            class_name = names.get(cls_id, "sports ball") if isinstance(names, dict) else str(cls_id)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": round(conf, 4),
                "class_id": cls_id,
                "class_name": class_name,
                "center": (cx, cy)
            })

        # Sort detections by confidence descending so highest confidence ball is first
        detections.sort(key=lambda d: d["conf"], reverse=True)
        return detections
