"""
app.py
Sports Ball Detection & Tracking System - Streamlit Dashboard
Student ID: 23341A4255 | Name: KONDRI AKHIL KRISHNA
Project: 37 — Sports Ball Detection & Tracking System | Model: YOLO11
"""

import os
import time
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Import internal modular components with graceful error handling
try:
    from src.detector import BallDetector, get_device
    from src.tracker import BallTracker
    from src.trajectory import BallTrajectory
    from src.analytics import VideoAnalytics
    from src.video_processor import VideoProcessor
    CV_PIPELINE_AVAILABLE = True
    CV_IMPORT_ERROR = None
except Exception as _cv_err:
    CV_PIPELINE_AVAILABLE = False
    CV_IMPORT_ERROR = str(_cv_err)

    def get_device():
        return "cpu", "Device: CPU (Awaiting dependencies)"


# Page configuration
st.set_page_config(
    page_title="Sports Ball Detection & Tracking | YOLO11",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    .student-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        border: 1px solid #334155;
        padding: 12px 16px;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #10b981;
    }
</style>
""", unsafe_allow_html=True)


def ensure_directories():
    """Ensure working directories exist."""
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)


@st.cache_resource(show_spinner="Loading YOLO11 neural weights...")
def get_cached_detector(model_name_or_path: str, use_custom: bool, custom_path: str, device: str) -> BallDetector:
    """Cache YOLO11 detector in memory across Streamlit interactions."""
    return BallDetector(
        model_name_or_path=model_name_or_path,
        use_custom_model=use_custom,
        custom_model_path=custom_path,
        device=device
    )


def create_synthetic_demo_video(output_path: str = "uploads/sample_soccer_demo.mp4") -> str:
    """
    Generate a synthetic sports clip with realistic physics of a bouncing soccer ball
    on a green grass pitch. Useful for immediate testing and demonstration.
    """
    if os.path.exists(output_path):
        return output_path

    ensure_directories()
    width, height = 960, 540
    fps = 30
    total_frames = 120  # 4 seconds

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Ball physics simulation
    ball_x = 80.0
    ball_y = 150.0
    vx = 7.5
    vy = -2.0
    gravity = 0.55
    bounce_damping = 0.78
    ground_y = height - 70
    radius = 18

    for f in range(total_frames):
        # Green pitch background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (34, 139, 34)  # Forest green

        # Pitch lines
        cv2.line(frame, (0, ground_y + radius), (width, ground_y + radius), (255, 255, 255), 3)
        cv2.circle(frame, (width // 2, height // 2), 70, (255, 255, 255), 2)
        cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)

        # Update physics
        vy += gravity
        ball_x += vx
        ball_y += vy

        if ball_y + radius >= ground_y:
            ball_y = ground_y - radius
            vy = -vy * bounce_damping

        if ball_x + radius >= width or ball_x - radius <= 0:
            vx = -vx

        # Draw sports ball (white with classic black pentagon pattern)
        bx, by = int(ball_x), int(ball_y)
        cv2.circle(frame, (bx, by), radius, (245, 245, 245), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (bx, by), radius, (20, 20, 20), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, (bx, by), 5, (20, 20, 20), -1, lineType=cv2.LINE_AA)
        for angle in range(0, 360, 72):
            rad = np.deg2rad(angle)
            px = int(bx + (radius - 5) * np.cos(rad))
            py = int(by + (radius - 5) * np.sin(rad))
            cv2.line(frame, (bx, by), (px, py), (40, 40, 40), 1, lineType=cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    return output_path


def main():
    ensure_directories()
    device_key, device_display = get_device()

    # Header section
    st.markdown('<div class="main-header">⚽ Sports Ball Detection & Tracking</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detect, track, and visualize sports ball movement across frames using YOLO11.</div>', unsafe_allow_html=True)

    # Student metadata card
    st.markdown(f"""
    <div class="student-card">
        <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;">
            <div>
                <strong style="color: #38bdf8;">Student ID:</strong> <span style="color: #f1f5f9;">23341A4255</span> &nbsp;|&nbsp;
                <strong style="color: #38bdf8;">Name:</strong> <span style="color: #f1f5f9;">KONDRI AKHIL KRISHNA</span> &nbsp;|&nbsp;
                <strong style="color: #38bdf8;">Project:</strong> <span style="color: #f1f5f9;">Project 37 — Sports Ball Detection & Tracking</span>
            </div>
            <div>
                <span style="background: #0284c7; color: white; padding: 3px 10px; border-radius: 6px; font-size: 0.85rem; font-weight: 600;">Model: YOLO11</span>
                &nbsp;
                <span style="background: #16a34a; color: white; padding: 3px 10px; border-radius: 6px; font-size: 0.85rem; font-weight: 600;">{device_display}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Detection & Tracking Configuration
    st.sidebar.header("⚙️ Detection Settings")

    # 1. Model Selection
    model_choice = st.sidebar.selectbox(
        "Model",
        options=["YOLO11n", "YOLO11s", "YOLO11m"],
        index=0,
        help="Select the YOLO11 model variant. YOLO11n is optimized for real-time speed."
    )
    model_file_map = {
        "YOLO11n": "yolo11n.pt",
        "YOLO11s": "yolo11s.pt",
        "YOLO11m": "yolo11m.pt"
    }
    selected_model_file = model_file_map[model_choice]

    # Custom model option
    use_custom_model = st.sidebar.checkbox("Use Custom Ball Model", value=False)
    custom_model_path = "models/best.pt"

    if use_custom_model:
        st.sidebar.info(f"Target custom model path: `{custom_model_path}`")
        if not os.path.exists(custom_model_path):
            st.sidebar.warning(f"⚠️ `{custom_model_path}` not found. You can upload custom weights below or train via `scripts/train.py`.")
            custom_upload = st.sidebar.file_uploader("Upload .pt model", type=["pt"], key="custom_pt")
            if custom_upload is not None:
                with open(custom_model_path, "wb") as f:
                    f.write(custom_upload.getbuffer())
                st.sidebar.success("Custom model uploaded successfully!")
        else:
            st.sidebar.success("✅ Custom ball model detected in `models/best.pt`.")

    # 2. Confidence Threshold
    conf_thresh = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05,
        help="Minimum confidence score required to detect a ball."
    )

    # 3. IoU Threshold
    iou_thresh = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05,
        help="Intersection over Union threshold for Non-Maximum Suppression."
    )

    # 4. Tracker
    tracker_choice = st.sidebar.selectbox(
        "Tracker",
        options=["ByteTrack", "BoT-SORT"],
        index=0,
        help="Tracking algorithm. ByteTrack is lightweight, simple, and reliable for fast objects."
    )

    # 5. Trajectory settings
    enable_trajectory = st.sidebar.checkbox("Enable trajectory", value=True)
    full_trajectory = st.sidebar.checkbox("Full trajectory (unlimited trail)", value=False)

    if full_trajectory:
        trajectory_length = 0
        st.sidebar.caption("Trailing: Entire video trajectory")
    else:
        trajectory_length = st.sidebar.slider(
            "Trajectory length (frames)",
            min_value=10,
            max_value=300,
            value=100,
            step=10,
            help="Number of historical frames to render as the ball's movement trail."
        )

    # Processing options
    with st.sidebar.expander("Advanced Options", expanded=False):
        imgsz = st.select_slider("Inference Image Size", options=[320, 480, 640, 960], value=640)
        display_preview_interval = st.slider("Preview Update Frequency (frames)", min_value=1, max_value=10, value=2)
        max_frames_choice = st.selectbox(
            "Processing Limit (Fast Cloud/CPU Testing)",
            options=["All Frames (Full Video)", "150 Frames (~5 sec clip)", "300 Frames (~10 sec clip)", "600 Frames (~20 sec clip)"],
            index=0,
            help="Select 'All Frames' for complete video processing, or limit frames for faster testing."
        )
        max_frames = None
        if "150" in max_frames_choice:
            max_frames = 150
        elif "300" in max_frames_choice:
            max_frames = 300
        elif "600" in max_frames_choice:
            max_frames = 600

    # Main Dashboard Tabs
    tab_home, tab_process, tab_analytics, tab_guide = st.tabs([
        "🏠 Video Input & Overview",
        "⚡ Detection & Tracking",
        "📊 Analytics & Movement",
        "📖 Project Documentation"
    ])

    # Manage session state for uploaded video
    if "current_video_path" not in st.session_state:
        st.session_state.current_video_path = None
    if "video_info" not in st.session_state:
        st.session_state.video_info = None
    if "analytics_result" not in st.session_state:
        st.session_state.analytics_result = None
    if "output_video_path" not in st.session_state:
        st.session_state.output_video_path = None

    # TAB 1: Video Input & Overview
    with tab_home:
        st.subheader("1. Video Input")
        st.write("Upload a sports video such as football, cricket, tennis, basketball, or baseball.")

        col_upload, col_demo = st.columns([3, 1])

        with col_upload:
            uploaded_file = st.file_uploader(
                "Choose sports footage",
                type=["mp4", "avi", "mov", "mkv"],
                help="Supported formats: MP4, AVI, MOV, MKV"
            )

        with col_demo:
            st.write("Or try demo video:")
            if st.button("⚽ Load Sample Clip", use_container_width=True):
                demo_path = create_synthetic_demo_video()
                st.session_state.current_video_path = demo_path
                st.session_state.video_info = VideoProcessor.get_video_info(demo_path)
                st.session_state.analytics_result = None
                st.session_state.output_video_path = None
                st.success("Sample soccer footage loaded!")

        if uploaded_file is not None:
            # Save uploaded video to uploads directory
            safe_filename = uploaded_file.name
            saved_path = os.path.join("uploads", safe_filename)
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                vinfo = VideoProcessor.get_video_info(saved_path)
                st.session_state.current_video_path = saved_path
                st.session_state.video_info = vinfo
                st.session_state.analytics_result = None
                st.session_state.output_video_path = None
            except Exception as e:
                st.error(f"Unable to read the uploaded video: {str(e)}")

        # Display video information and preview if available
        if st.session_state.current_video_path and st.session_state.video_info:
            vinfo = st.session_state.video_info
            st.markdown("### 📋 Video Information")

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Filename", vinfo["filename"])
            with c2:
                st.metric("Resolution", vinfo["resolution"])
            with c3:
                st.metric("FPS", f"{vinfo['fps']}")
            with c4:
                st.metric("Duration", f"{vinfo['duration']}s ({vinfo['duration_formatted']})")
            with c5:
                st.metric("Total Frames", f"{vinfo['total_frames']}")

            col_vid, col_info = st.columns([2, 1])
            with col_vid:
                st.video(st.session_state.current_video_path)
            with col_info:
                st.info("""
                **Next Steps:**
                1. Switch to the **Detection & Tracking** tab.
                2. Adjust model, confidence, and tracker settings in the sidebar.
                3. Click **Start Detection & Tracking** to process the video.
                """)
        else:
            st.info("Please upload a sports video to begin.")

    # TAB 2: Detection & Tracking
    with tab_process:
        st.subheader("2. Real-Time Ball Detection & Tracking")

        if not st.session_state.current_video_path:
            st.warning("Please upload a sports video in the Video Input tab first.")
        else:
            col_ctrl1, col_ctrl2 = st.columns([2, 1])
            with col_ctrl1:
                st.write(f"**Target Video:** `{os.path.basename(st.session_state.current_video_path)}`")
            with col_ctrl2:
                start_btn = st.button("🚀 Start Detection & Tracking", type="primary", use_container_width=True)

            # Execution container
            if start_btn:
                if not CV_PIPELINE_AVAILABLE:
                    st.error(f"Cannot initialize computer vision pipeline: Missing dependency ({CV_IMPORT_ERROR}). Please install requirements: pip install -r requirements.txt")
                    st.stop()

                # Validate model file
                if use_custom_model and not os.path.exists(custom_model_path):
                    st.error(f"Custom model file '{custom_model_path}' is missing. Please place weights in `models/` or uncheck 'Use Custom Ball Model'.")
                    st.stop()

                # Initialize Detector, Tracker, Trajectory, Processor
                with st.spinner("Initializing YOLO11 Ball Detector and Tracker..."):
                    try:
                        detector = get_cached_detector(
                            model_name_or_path=selected_model_file,
                            use_custom=use_custom_model,
                            custom_path=custom_model_path,
                            device=device_key
                        )
                        tracker = BallTracker(
                            model=detector.model,
                            tracker_type=tracker_choice,
                            ball_class_ids=detector.ball_class_ids,
                            device=device_key
                        )
                        trajectory = BallTrajectory(
                            max_length=trajectory_length,
                            enabled=enable_trajectory
                        )
                        processor = VideoProcessor(
                            detector=detector,
                            tracker=tracker,
                            trajectory=trajectory,
                            output_dir="output"
                        )
                    except Exception as e:
                        st.error(f"Error during initialization: {str(e)}")
                        st.stop()

                st.info(f"Processing with **{detector.model_path}** on **{detector.device_display}** using **{tracker_choice}**...")

                # Progress placeholders
                progress_bar = st.progress(0)
                status_text = st.empty()

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                stat_frame = m_col1.empty()
                stat_fps = m_col2.empty()
                stat_ball = m_col3.empty()
                stat_time = m_col4.empty()

                preview_placeholder = st.empty()

                start_proc_time = time.time()

                # UI progress callback
                def on_frame_processed(cur_f, tot_f, cur_fps, ann_frame, track_info):
                    pct = cur_f / tot_f if tot_f > 0 else 0.0
                    progress_bar.progress(min(pct, 1.0))

                    elapsed = time.time() - start_proc_time
                    est_total = (elapsed / cur_f * tot_f) if cur_f > 0 else 0
                    remaining = max(0, est_total - elapsed)

                    status_text.text(f"Processing: Frame {cur_f} of {tot_f} ({int(pct * 100)}%)")

                    stat_frame.metric("Frame", f"{cur_f} / {tot_f}")
                    stat_fps.metric("Processing FPS", f"{cur_fps:.1f}")
                    if track_info:
                        stat_ball.metric("Ball ID", f"{track_info['track_id']} (Conf: {track_info['conf']:.2f})")
                    else:
                        stat_ball.metric("Ball ID", "Not Detected")
                    stat_time.metric("Est. Remaining", f"{int(remaining)}s")

                    if cur_f % display_preview_interval == 0:
                        # Convert BGR to RGB for Streamlit preview
                        rgb_frame = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
                        preview_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                try:
                    out_path, analytics = processor.process_video(
                        video_path=st.session_state.current_video_path,
                        conf_threshold=conf_thresh,
                        iou_threshold=iou_thresh,
                        imgsz=imgsz,
                        output_filename="tracked_video.mp4",
                        max_frames=max_frames,
                        progress_callback=on_frame_processed
                    )
                    st.session_state.output_video_path = out_path
                    st.session_state.analytics_result = analytics

                    progress_bar.progress(1.0)
                    status_text.text("Processing Complete!")
                    st.success("🎉 Video processed successfully! View results below.")
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

            # Display Processed Output Video if ready
            if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
                st.markdown("---")
                st.markdown("### 🎬 Processed Video Output")

                col_v, col_d = st.columns([3, 1])
                with col_v:
                    # Read binary for download and preview
                    with open(st.session_state.output_video_path, "rb") as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes)

                with col_d:
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=video_bytes,
                        file_name="tracked_sports_ball.mp4",
                        mime="video/mp4",
                        type="primary",
                        use_container_width=True
                    )
                    st.info("Annotations included:\n- Bounding box\n- Ball ID & Confidence\n- Center point & Trajectory\n- Real-time HUD Telemetry")

    # TAB 3: Analytics & Movement
    with tab_analytics:
        st.subheader("3. Actual Detection & Movement Analytics")

        if st.session_state.analytics_result is None:
            st.info("No analytics available yet. Please run video processing in the 'Detection & Tracking' tab to view actual performance metrics.")
        else:
            analytics: VideoAnalytics = st.session_state.analytics_result
            summary = analytics.compute_summary()

            if summary["detected_frames"] == 0:
                st.warning("⚠️ No ball detected in the current video. Try lowering the Confidence Threshold in the sidebar or using a custom-trained model.")

            st.markdown("#### 🎯 Detection Statistics")
            r1c1, r1c2, r1c3, r1c4, r1c5, r1c6 = st.columns(6)
            r1c1.metric("Total Frames", summary["total_frames"])
            r1c2.metric("Detected Frames", summary["detected_frames"])
            r1c3.metric("Missed Frames", summary["missed_frames"])
            r1c4.metric("Detection Rate", f"{summary['detection_rate']}%")
            r1c5.metric("Avg Confidence", f"{summary['avg_confidence']:.2f}")
            r1c6.metric("Max Confidence", f"{summary['max_confidence']:.2f}")

            st.markdown("#### 🔄 Tracking Statistics")
            r2c1, r2c2, r2c3 = st.columns(3)
            r2c1.metric("Unique Tracking IDs", summary["unique_track_ids"])
            r2c2.metric("Tracking Duration", f"{summary['tracking_duration_sec']}s ({summary['tracking_duration_frames']} frames)")
            r2c3.metric("Trajectory Points", summary["trajectory_points"])

            st.markdown("#### 📏 Ball Movement Analysis (Pixel-Based)")
            st.caption("ℹ️ Note: Measurements are calculated in image coordinate pixel space. Real-world speed (e.g. km/h) requires specific camera calibration and pitch homography.")
            r3c1, r3c2, r3c3 = st.columns(3)
            r3c1.metric("Total Pixel Displacement", f"{summary['total_pixel_displacement']} px")
            r3c2.metric("Avg Movement / Frame", f"{summary['avg_movement_per_frame']} px")
            r3c3.metric("Max Movement / Frame", f"{summary['max_movement_per_frame']} px")

            # Charts
            st.markdown("---")
            st.markdown("### 📈 Visualizations")

            g_col1, g_col2 = st.columns(2)
            with g_col1:
                traj_fig = analytics.plot_trajectory_2d()
                if traj_fig:
                    st.pyplot(traj_fig)
                else:
                    st.info("Insufficient trajectory points to generate 2D path chart.")

            with g_col2:
                conf_fig = analytics.plot_confidence_timeline()
                if conf_fig:
                    st.pyplot(conf_fig)

            g_col3, g_col4 = st.columns(2)
            with g_col3:
                avail_fig = analytics.plot_detection_availability()
                if avail_fig:
                    st.pyplot(avail_fig)

            with g_col4:
                mov_fig = analytics.plot_movement_analysis()
                if mov_fig:
                    st.pyplot(mov_fig)

            # Export data
            st.markdown("---")
            csv_data = analytics.to_csv_string()
            if csv_data:
                st.download_button(
                    label="📥 Export Frame-by-Frame Analytics (CSV)",
                    data=csv_data,
                    file_name="ball_tracking_analytics.csv",
                    mime="text/csv",
                    use_container_width=False
                )

    # TAB 4: Project Documentation & Architecture
    with tab_guide:
        st.subheader("4. Project Architecture & Documentation")
        st.markdown(f"""
        ### Student & Project Details
        - **Student Name:** KONDRI AKHIL KRISHNA
        - **Student ID:** 23341A4255
        - **Project Title:** Project 37 — Sports Ball Detection & Tracking System
        - **Architecture:** YOLO11 Detection + ByteTrack/BoT-SORT + Center Trajectory Estimation

        ### Pipeline Architecture
        ```text
        Upload Video (MP4 / AVI / MOV / MKV)
                 ↓
        Frame Extraction (OpenCV)
                 ↓
        YOLO11 Detection (yolo11n / custom best.pt)
                 ↓
        Ball Class Filtering (COCO Class 32 / Custom Class 0)
                 ↓
        Object Tracking (ByteTrack / BoT-SORT)
                 ↓
        Center Coordinate Calculation & Trajectory Trail
                 ↓
        Displacement & Instantaneous Speed Calculation
                 ↓
        Video Annotation & Telemetry HUD Overlay
                 ↓
        Final Output Video & Comprehensive Analytics
        ```

        ### How to Train a Custom Ball Model
        1. Place your dataset in `dataset/` following YOLO structure:
           - `dataset/images/train`, `dataset/images/val`, `dataset/images/test`
           - `dataset/labels/train`, `dataset/labels/val`, `dataset/labels/test`
           - `dataset/data.yaml`
        2. Run the training script:
           ```bash
           python scripts/train.py --model yolo11n.pt --epochs 50 --batch 16
           ```
        3. The fine-tuned weights will be saved directly to `models/best.pt`.
        4. Enable **"Use Custom Ball Model"** in the Streamlit sidebar.

        ### Technical Limitations
        - **Small Object Scale:** At long-range camera shots, sports balls occupy very few pixels ($< 10 \\times 10$ px).
        - **Motion Blur:** High-velocity ball movement can cause severe blur, reducing detection confidence.
        - **Occlusions:** Players blocking the ball temporarily interrupt tracking; the tracker handles re-identification.
        - **Calibration:** Speed is reported in pixel displacement; converting to real-world velocity requires fixed camera calibration.
        """)


if __name__ == "__main__":
    main()
