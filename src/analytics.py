"""
analytics.py
Calculates detection, tracking, and pixel-based movement metrics strictly
from actual video processing results, and produces publication-quality charts.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# Set clean non-interactive matplotlib backend
matplotlib.use("Agg")


class VideoAnalytics:
    """
    Computes rigorous analytical metrics and visualizations from actual ball tracking history.
    """

    def __init__(self, history: List[Dict[str, Any]], fps: float = 30.0, video_width: int = 1920, video_height: int = 1080):
        """
        Args:
            history: List of frame-by-frame detection records from BallTrajectory.history
            fps: Video frames per second
            video_width: Video frame width in pixels
            video_height: Video frame height in pixels
        """
        self.history = history
        self.fps = fps if fps > 0 else 30.0
        self.width = video_width
        self.height = video_height
        self.df = pd.DataFrame(history) if history else pd.DataFrame()

    def compute_summary(self) -> Dict[str, Any]:
        """
        Compute detection, tracking, and pixel movement statistics.
        All values are derived directly from the actual processed video detections.
        """
        if self.df.empty:
            return {
                "total_frames": 0,
                "detected_frames": 0,
                "missed_frames": 0,
                "detection_rate": 0.0,
                "avg_confidence": 0.0,
                "max_confidence": 0.0,
                "unique_track_ids": 0,
                "tracking_duration_frames": 0,
                "tracking_duration_sec": 0.0,
                "trajectory_points": 0,
                "total_pixel_displacement": 0.0,
                "avg_movement_per_frame": 0.0,
                "max_movement_per_frame": 0.0
            }

        total_frames = len(self.df)
        detected_mask = self.df["x"].notnull()
        detected_df = self.df[detected_mask]

        detected_frames = int(detected_mask.sum())
        missed_frames = total_frames - detected_frames
        detection_rate = (detected_frames / total_frames * 100.0) if total_frames > 0 else 0.0

        if detected_frames > 0:
            avg_confidence = float(detected_df["conf"].mean())
            max_confidence = float(detected_df["conf"].max())
            unique_ids = int(detected_df["track_id"].nunique())
            trajectory_points = detected_frames

            # Movement metrics (pixel-based)
            movements = detected_df["movement"].dropna().tolist()
            # Filter non-zero movements for meaningful averages if desired, but standard displacement is sum:
            total_displacement = float(np.sum(movements))
            avg_movement = float(np.mean(movements)) if movements else 0.0
            max_movement = float(np.max(movements)) if movements else 0.0
            tracking_duration_frames = detected_frames
            tracking_duration_sec = detected_frames / self.fps
        else:
            avg_confidence = 0.0
            max_confidence = 0.0
            unique_ids = 0
            trajectory_points = 0
            total_displacement = 0.0
            avg_movement = 0.0
            max_movement = 0.0
            tracking_duration_frames = 0
            tracking_duration_sec = 0.0

        return {
            "total_frames": total_frames,
            "detected_frames": detected_frames,
            "missed_frames": missed_frames,
            "detection_rate": round(detection_rate, 2),
            "avg_confidence": round(avg_confidence, 4),
            "max_confidence": round(max_confidence, 4),
            "unique_track_ids": unique_ids,
            "tracking_duration_frames": tracking_duration_frames,
            "tracking_duration_sec": round(tracking_duration_sec, 2),
            "trajectory_points": trajectory_points,
            "total_pixel_displacement": round(total_displacement, 2),
            "avg_movement_per_frame": round(avg_movement, 2),
            "max_movement_per_frame": round(max_movement, 2)
        }

    def plot_trajectory_2d(self) -> Optional[plt.Figure]:
        """
        Generate 2D Ball Trajectory Plot (X coordinate vs Y coordinate).
        Inverted Y-axis to match natural video image coordinate system (origin top-left).
        """
        if self.df.empty:
            return None

        detected = self.df.dropna(subset=["x", "y"])
        if len(detected) < 2:
            return None

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#1f2937")

        # Color lines by timeline progression
        x = detected["x"].values
        y = detected["y"].values
        frames = detected["frame"].values

        scatter = ax.scatter(
            x, y,
            c=frames,
            cmap="viridis",
            s=25,
            edgecolors="none",
            alpha=0.85,
            zorder=3,
            label="Trajectory Point"
        )
        ax.plot(x, y, color="#10b981", alpha=0.5, linewidth=1.5, zorder=2)

        # Highlight Start and End points
        ax.scatter(x[0], y[0], color="#3b82f6", s=120, edgecolors="white", linewidth=2, label="Start", zorder=4)
        ax.scatter(x[-1], y[-1], color="#ef4444", s=120, edgecolors="white", linewidth=2, label="End", zorder=4)

        # Match video frame coordinates
        ax.set_xlim(0, max(self.width, int(max(x) * 1.05)))
        ax.set_ylim(max(self.height, int(max(y) * 1.05)), 0)  # Invert Y to match image coordinates

        ax.set_title("Ball Trajectory (2D Video Coordinate Space)", color="#f9fafb", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("X Coordinate (pixels)", color="#e5e7eb", fontsize=11)
        ax.set_ylabel("Y Coordinate (pixels)", color="#e5e7eb", fontsize=11)
        ax.tick_params(colors="#9ca3af")
        ax.grid(True, linestyle="--", alpha=0.25, color="#6b7280")

        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Frame Number", color="#e5e7eb", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="#9ca3af")
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#9ca3af")

        ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#f9fafb")
        plt.tight_layout()
        return fig

    def plot_confidence_timeline(self) -> Optional[plt.Figure]:
        """
        Generate Confidence Over Time plot (Frame Number vs Confidence Score).
        """
        if self.df.empty:
            return None

        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#1f2937")

        frames = self.df["frame"].values
        conf = self.df["conf"].values

        ax.plot(frames, conf, color="#06b6d4", linewidth=1.8, label="Ball Detection Confidence")
        ax.fill_between(frames, conf, 0, color="#06b6d4", alpha=0.2)

        # Baseline average confidence
        detected = self.df[self.df["conf"] > 0]
        if not detected.empty:
            avg_conf = detected["conf"].mean()
            ax.axhline(avg_conf, color="#f59e0b", linestyle="--", linewidth=1.2, label=f"Mean Confidence ({avg_conf:.2f})")

        ax.set_ylim(0, 1.05)
        ax.set_title("Detection Confidence Over Video Timeline", color="#f9fafb", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Frame Number", color="#e5e7eb", fontsize=11)
        ax.set_ylabel("Confidence Score", color="#e5e7eb", fontsize=11)
        ax.tick_params(colors="#9ca3af")
        ax.grid(True, linestyle="--", alpha=0.25, color="#6b7280")
        ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#f9fafb", loc="lower right")

        plt.tight_layout()
        return fig

    def plot_detection_availability(self) -> Optional[plt.Figure]:
        """
        Generate Detection Availability plot (Detected / Not Detected over time).
        """
        if self.df.empty:
            return None

        fig, ax = plt.subplots(figsize=(8, 2.8), dpi=150)
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#1f2937")

        frames = self.df["frame"].values
        is_detected = (self.df["x"].notnull()).astype(int).values

        # Plot binary bar/stem
        ax.bar(frames, is_detected, color=np.where(is_detected == 1, "#10b981", "#ef4444"), width=1.0, alpha=0.85)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Not Detected", "Detected"], color="#e5e7eb", fontsize=10)
        ax.set_ylim(-0.1, 1.2)
        ax.set_title("Ball Detection Availability Across Frames", color="#f9fafb", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Frame Number", color="#e5e7eb", fontsize=11)
        ax.tick_params(colors="#9ca3af")
        ax.grid(True, linestyle="--", alpha=0.2, color="#6b7280", axis="x")

        plt.tight_layout()
        return fig

    def plot_movement_analysis(self) -> Optional[plt.Figure]:
        """
        Generate Instantaneous Ball Movement (pixels/frame) over time.
        """
        if self.df.empty:
            return None

        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#1f2937")

        frames = self.df["frame"].values
        movement = self.df["movement"].fillna(0).values

        ax.plot(frames, movement, color="#a855f7", linewidth=1.6, label="Displacement (pixels/frame)")
        ax.fill_between(frames, movement, 0, color="#a855f7", alpha=0.15)

        ax.set_title("Ball Movement Magnitude (Pixel-Based Measurement)", color="#f9fafb", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Frame Number", color="#e5e7eb", fontsize=11)
        ax.set_ylabel("Displacement (pixels)", color="#e5e7eb", fontsize=11)
        ax.tick_params(colors="#9ca3af")
        ax.grid(True, linestyle="--", alpha=0.25, color="#6b7280")
        ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#f9fafb")

        plt.tight_layout()
        return fig

    def to_csv_string(self) -> str:
        """Convert history dataframe to CSV string for download."""
        if self.df.empty:
            return ""
        return self.df.to_csv(index=False)
