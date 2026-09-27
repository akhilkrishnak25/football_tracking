# Project 37 — Sports Ball Detection & Tracking System

## Student Details

* **Student ID:** 23341A4255
* **Name:** KONDRI AKHIL KRISHNA
* **Project:** Sports Ball Detection & Tracking
* **Model:** YOLO11

---

## 📌 Project Overview

The **Sports Ball Detection and Tracking System** is an end-to-end computer vision application built with **YOLO11**, **ByteTrack / BoT-SORT**, and **Streamlit**. It processes sports footage (such as football, cricket, tennis, basketball, and baseball), detects the sports ball on a frame-by-frame basis, persistently tracks the ball across time, generates a smooth dynamic movement trajectory trail, overlays real-time telemetry HUD metrics, and outputs an annotated MP4 video along with real analytical reports.

The application is structured for academic project demonstration, high reproducibility, and practical real-world execution.

---

## 🚀 Key Features

* **Video Ingestion & Inspection:**
  * Supports MP4, AVI, MOV, and MKV video formats.
  * Automatic metadata extraction: Filename, Resolution, FPS, Duration, and Total Frame Count.
  * Built-in sample video generator for immediate demonstration.
* **YOLO11 Ball Detection:**
  * Pre-trained YOLO11 support: **YOLO11n**, **YOLO11s**, and **YOLO11m**.
  * Intelligent class filtering strictly for sports balls (COCO class 32) or custom-trained ball classes.
  * Configurable Confidence and IoU thresholds.
* **Persistent Multi-Object Tracking:**
  * Seamless integration with **ByteTrack** and **BoT-SORT**.
  * Persistent tracking ID assignment (`Ball ID: 1`) across consecutive frames.
  * Handles fast motion, player occlusions, and brief detection gaps.
* **Visual Ball Trajectory:**
  * Automatic center coordinate calculation: $\left(\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2}\right)$.
  * Dynamic gradient trajectory trail (with configurable length from 10 to 300 frames or full trajectory).
* **Real-Time HUD Telemetry:**
  * Displays ball ID, confidence, center pixel coordinates, instantaneous processing FPS, and frame count.
* **Actual Detection Analytics & Movement Analysis:**
  * Strictly calculated from actual detections (zero fake/mock data).
  * Detection rate, average/max confidence, tracking duration, and unique tracking IDs.
  * Pixel displacement and average/maximum pixel speed per frame.
  * Interactive 2D trajectory path plot, confidence timeline, detection availability chart, and movement magnitude graph.
  * CSV analytics export.
* **Custom Model & Dataset Training:**
  * Ready-to-use dataset directory structure and `data.yaml`.
  * Dedicated training script `scripts/train.py` to train YOLO11 on custom ball datasets and save weights to `models/best.pt`.

---

## 🛠️ Technologies Used

* **Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch & Torchvision
* **Object Detection & Tracking:** Ultralytics YOLO11, ByteTrack, BoT-SORT
* **Computer Vision:** OpenCV (cv2), Pillow
* **Data Processing & Analytics:** NumPy, Pandas
* **Visualization:** Matplotlib
* **Web Interface:** Streamlit

---

## 📁 Project Structure

```text
football/
│
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git ignore rules
│
├── models/                     # Custom trained weights directory
│   ├── best.pt                 # Target location for custom YOLO11 model
│   └── README.md
│
├── dataset/                    # Custom sports ball dataset directory
│   ├── images/
│   │   ├── train/              # Training images
│   │   ├── val/                # Validation images
│   │   └── test/               # Test images
│   ├── labels/
│   │   ├── train/              # YOLO format annotations (.txt)
│   │   ├── val/
│   │   └── test/
│   └── data.yaml               # YOLO11 dataset configuration file
│
├── src/                        # Modular source code package
│   ├── __init__.py
│   ├── detector.py             # YOLO11 ball detector & device manager
│   ├── tracker.py              # ByteTrack & BoT-SORT tracking manager
│   ├── trajectory.py           # Center tracking, trajectory trail & movement
│   ├── analytics.py            # Real statistics computation & plotting
│   └── video_processor.py      # Video decoding, pipeline orchestration & encoding
│
├── scripts/
│   └── train.py                # YOLO11 custom training script
│
├── uploads/                    # Directory for uploaded input videos
└── output/                     # Directory for generated annotated videos
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open the Project
Ensure you are in the project root directory:
```bash
cd football
```

### 2. Create and Activate a Virtual Environment
**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
Once started, the application will open automatically in your browser at `http://localhost:8501`.

---

## 📖 Step-by-Step Usage Flow

1. **Video Ingestion:**
   * Open the **"Video Input & Overview"** tab.
   * Upload an MP4, AVI, MOV, or MKV sports video, or click **"⚽ Load Sample Clip"** to immediately test with the built-in soccer simulation.
   * View the video details (resolution, FPS, duration, total frames).
2. **Configure Settings (Sidebar):**
   * Select YOLO11 model (`YOLO11n`, `YOLO11s`, `YOLO11m`).
   * Optionally check **"Use Custom Ball Model"** to load `models/best.pt`.
   * Adjust **Confidence Threshold** (e.g. `0.40 - 0.50`) and **IoU Threshold** (e.g. `0.50`).
   * Select tracking algorithm: **ByteTrack** or **BoT-SORT**.
   * Toggle trajectory trail and select trajectory trail length (e.g. `100 frames`).
3. **Run Detection & Tracking:**
   * Navigate to the **"Detection & Tracking"** tab.
   * Click **"🚀 Start Detection & Tracking"**.
   * Watch the real-time live preview showing the ball's bounding box, persistent ID, trajectory, and HUD overlay.
4. **Download Annotated Video:**
   * Once finished, preview the tracked video directly in the web player.
   * Click **"⬇️ Download Processed Video"** to save `tracked_sports_ball.mp4`.
5. **Analyze Movement & Performance:**
   * Switch to the **"Analytics & Movement"** tab.
   * Review detection rates, average confidence, tracking duration, and pixel displacement.
   * Inspect 2D ball trajectory path, confidence over time, and displacement per frame.
   * Download the complete frame-by-frame data via **"📥 Export Frame-by-Frame Analytics (CSV)"**.

---

## 🧠 Custom Ball Model Training

If you wish to fine-tune YOLO11 on your own sports ball dataset:

1. Place your dataset images and YOLO annotation labels into `dataset/`:
   ```text
   dataset/
   ├── images/train/, images/val/, images/test/
   ├── labels/train/, labels/val/, labels/test/
   └── data.yaml
   ```
2. Verify `dataset/data.yaml`:
   ```yaml
   path: ./dataset
   train: images/train
   val: images/val
   test: images/test
   names:
     0: ball
   ```
3. Run the training script:
   ```bash
   python scripts/train.py --model yolo11n.pt --epochs 50 --batch 16 --imgsz 640
   ```
4. The best model weights will be saved to `models/best.pt`.
5. Check **"Use Custom Ball Model"** in the Streamlit application to use your trained weights!

---

## ⚠️ Technical Limitations

1. **Extreme Object Scale:** In wide-angle broadcast shots, the sports ball may occupy very few pixels ($< 10 \times 10$ px), occasionally challenging detection without custom high-resolution fine-tuning.
2. **Severe Motion Blur:** At high ball velocities or lower shutter speeds, motion blur can distort the ball's contour and decrease confidence.
3. **Occlusions:** Physical occlusion by players, goalkeepers, or goalposts can temporarily obstruct the ball. ByteTrack maintains track associations during brief interruptions.
4. **Camera Motion vs. Absolute Trajectory:** Trajectory coordinates are recorded relative to the video image frame. If the broadcast camera pans or tilts, the 2D pixel trajectory reflects both camera movement and ball movement.
5. **Pixel-Based Measurement:** Movement is quantified in pixels ($\Delta \text{px} = \sqrt{\Delta x^2 + \Delta y^2}$). Converting to real-world velocity (e.g. km/h) requires specific pitch homography and camera calibration.

---

## 👨‍🎓 Academic Presentation Checklist

- [x] YOLO11 object detection integrated
- [x] ByteTrack / BoT-SORT persistent tracking implemented
- [x] Dynamic ball trajectory trail rendered
- [x] Live telemetry HUD overlay embedded on output video
- [x] Streamlit web dashboard with sidebar controls
- [x] Real-time live frame preview and progress monitoring
- [x] Accurate detection and movement analytics derived from actual video
- [x] Exportable annotated MP4 video and CSV telemetry
- [x] Custom dataset structure & training script included
- [x] Error handling for missing files, invalid videos, and undetected balls
