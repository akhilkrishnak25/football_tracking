"""
video_processor.py
Coordinates video decoding, YOLO11 inference, multi-object tracking,
HUD overlay composition, trajectory visualization, and output video encoding.
"""

import os
import time
from typing import Dict, Any, Generator, Optional, Tuple, Callable
import cv2
import numpy as np

from src.detector import BallDetector
from src.tracker import BallTracker
from src.trajectory import BallTrajectory
from src.analytics import VideoAnalytics


class VideoProcessor:
    """
    High-performance video processing pipeline for sports ball detection and tracking.
    """

    def __init__(
        self,
        detector: BallDetector,
        tracker: BallTracker,
        trajectory: BallTrajectory,
        output_dir: str = "output"
    ):
        self.detector = detector
        self.tracker = tracker
        self.trajectory = trajectory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """
        Extract video technical metadata.
        Returns:
            Dict containing filename, resolution, fps, duration, and total_frames.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file '{video_path}' does not exist.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to read the uploaded video: '{video_path}'.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = (total_frames / fps) if fps > 0 else 0.0

        cap.release()

        return {
            "filename": os.path.basename(video_path),
            "filepath": video_path,
            "width": width,
            "height": height,
            "resolution": f"{width} x {height}",
            "fps": round(fps, 2),
            "duration": round(duration_sec, 2),
            "duration_formatted": f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}",
            "total_frames": total_frames
        }

    def _draw_hud(
        self,
        frame: np.ndarray,
        track_info: Optional[Dict[str, Any]],
        fps: float,
        frame_idx: int,
        total_frames: int
    ) -> np.ndarray:
        """
        Draw a modern semi-transparent HUD telemetry overlay card on the frame.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # HUD box geometry (top-left)
        box_w, box_h = 320, 160
        x1, y1 = 20, 20
        x2, y2 = x1 + box_w, y1 + box_h

        # Semi-transparent dark background
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 24, 33), -1)
        # Subtle border
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (59, 130, 246), 1)

        alpha = 0.75
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Title
        cv2.putText(
            frame,
            "Ball Detection & Tracking",
            (x1 + 12, y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA
        )
        cv2.line(frame, (x1 + 12, y1 + 34), (x2 - 12, y1 + 34), (75, 85, 99), 1)

        if track_info:
            ball_id = track_info.get("track_id", "N/A")
            conf = track_info.get("conf", 0.0)
            cx, cy = track_info.get("center", (0, 0))

            line1 = f"Ball ID: {ball_id}"
            line2 = f"Confidence: {conf:.2f}"
            line3 = f"Center: ({cx}, {cy})"
            status_color = (74, 222, 128)  # Neon Green
        else:
            line1 = "Ball ID: Not Detected"
            line2 = "Confidence: --"
            line3 = "Center: --"
            status_color = (239, 68, 68)  # Red

        line4 = f"FPS: {fps:.1f}"
        line5 = f"Frame: {frame_idx} / {total_frames}"

        # Draw metrics
        cv2.putText(frame, line1, (x1 + 14, y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1, cv2.LINE_AA)
        cv2.putText(frame, line2, (x1 + 14, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, line3, (x1 + 14, y1 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, line4, (x1 + 14, y1 + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (56, 189, 248), 1, cv2.LINE_AA)
        cv2.putText(frame, line5, (x1 + 14, y1 + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (156, 163, 175), 1, cv2.LINE_AA)

        return frame

    def _draw_ball_annotation(self, frame: np.ndarray, track: Dict[str, Any]) -> np.ndarray:
        """
        Draw bounding box, label tag, and center point for a detected ball.
        """
        x1, y1, x2, y2 = track["bbox"]
        track_id = track.get("track_id", 1)
        conf = track["conf"]
        cx, cy = track["center"]

        # Color: vibrant neon green/cyan box
        box_color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, lineType=cv2.LINE_AA)

        # Draw center point
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        # Label tag above bounding box
        label = f"Ball ID: {track_id} ({conf:.2f})"
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tag_y = max(y1 - 6, h + 8)

        # Background tag rectangle
        cv2.rectangle(frame, (x1, tag_y - h - 4), (x1 + w + 8, tag_y + baseline), (0, 200, 0), -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, tag_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA
        )

        return frame

    def process_video(
        self,
        video_path: str,
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.50,
        imgsz: int = 640,
        output_filename: str = "tracked_video.mp4",
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, float, np.ndarray, Optional[Dict[str, Any]]], None]] = None
    ) -> Tuple[str, VideoAnalytics]:
        """
        Execute full end-to-end processing pipeline on the video.

        Args:
            video_path: Source sports video file path
            conf_threshold: YOLO confidence threshold
            iou_threshold: YOLO IoU threshold
            imgsz: Processing resolution
            output_filename: Output video filename
            progress_callback: Optional callback for real-time UI streaming:
                               (current_frame, total_frames, fps, annotated_frame, track_info)

        Returns:
            Tuple[output_video_path, VideoAnalytics]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to read the uploaded video: '{video_path}'.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        effective_total = min(total_frames, max_frames) if (max_frames and max_frames > 0) else total_frames

        output_path = os.path.join(self.output_dir, output_filename)

        # VideoWriter configuration with robust MP4 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Reset states
        self.tracker.reset()
        self.trajectory.clear()

        frame_idx = 0
        proc_start_time = time.time()
        fps_measure_time = time.time()
        current_fps = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                t0 = time.time()

                # Step 1: Track Ball in current frame
                tracks = self.tracker.track_frame(
                    frame=frame,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    imgsz=imgsz
                )

                best_track = None
                if tracks:
                    # Select highest-confidence tracked ball
                    best_track = tracks[0]
                    self.trajectory.add_point(
                        frame_idx=frame_idx,
                        center=best_track["center"],
                        track_id=best_track["track_id"],
                        confidence=best_track["conf"]
                    )
                else:
                    self.trajectory.record_missed_frame(frame_idx)

                # Step 2: Draw Trajectory
                annotated_frame = self.trajectory.draw_trajectory(frame.copy())

                # Step 3: Draw Bounding Box and ID
                if best_track:
                    annotated_frame = self._draw_ball_annotation(annotated_frame, best_track)

                # Calculate instantaneous FPS
                frame_duration = time.time() - t0
                if frame_duration > 0:
                    inst_fps = 1.0 / frame_duration
                    # Exponential smoothing for stable HUD FPS display
                    current_fps = 0.85 * current_fps + 0.15 * inst_fps if current_fps > 0 else inst_fps

                # Step 4: Draw Telemetry HUD Overlay
                annotated_frame = self._draw_hud(
                    frame=annotated_frame,
                    track_info=best_track,
                    fps=current_fps,
                    frame_idx=frame_idx,
                    total_frames=effective_total
                )

                # Step 5: Write to output video
                writer.write(annotated_frame)

                # Step 6: Trigger UI Progress Callback if provided
                if progress_callback:
                    progress_callback(frame_idx, effective_total, current_fps, annotated_frame, best_track)

                if max_frames and frame_idx >= max_frames:
                    break

        finally:
            cap.release()
            writer.release()

        # Build Analytics from actual detection history
        analytics = VideoAnalytics(
            history=self.trajectory.history,
            fps=fps,
            video_width=width,
            video_height=height
        )

        # Transcode to web-safe H.264 if imageio_ffmpeg is present
        final_video_path = output_path
        try:
            import imageio_ffmpeg
            import subprocess
            h264_path = output_path.replace(".mp4", "_h264.mp4")
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-y", "-i", output_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "fast", "-crf", "23", h264_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
                final_video_path = h264_path
        except Exception:
            pass

        return final_video_path, analytics
