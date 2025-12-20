# Tech Challenge Phase 4 - Video Analysis with AI

A comprehensive video analysis system that performs face detection, emotion analysis, pose estimation, and activity recognition using state-of-the-art AI models.

## Overview

This project implements a complete video analysis pipeline that:
1.  **Detects Faces & Emotions**: Uses DeepFace to identify dominant emotions (happy, sad, angry, etc.) in video frames.
2.  **Estimates Poses**: Leverages YOLOv11-pose and MediaPipe to detect human skeletons and keypoints.
3.  **Recognizes Activities**: Classifies body postures (standing, sitting, lying down) based on geometric analysis of keypoints.
4.  **Detects Anomalies**: Identifies sudden, high-velocity movements that might indicate unusual events.
5.  **Generates Reports**: Produces detailed JSON summaries with statistics on all detected events.

## Technologies

-   **Python 3.11**: Main programming language.
-   **OpenCV**: Video processing (frame reading/writing) and real-time webcam feed.
-   **YOLOv11 (Ultralytics)**: Advanced pose estimation and object detection.
    -   High accuracy keypoint detection.
    -   Robust against partial occlusions.
-   **DeepFace**: Facial attribute analysis (emotion recognition) using pre-trained CNNs.
-   **MediaPipe**: Alternative pose detection library (Google) used in specific modules.
-   **NumPy**: Efficient numerical operations and data processing.

## Project Structure

```
TC04/
├── combined_detection.py       # Main pipeline: Integrated Pose, Emotion, and Activity detection
├── detect_expression_video.py  # Dedicated script for emotion detection in videos
├── facil_detection.py          # Real-time face detection using webcam (Haar Cascades)
├── pose_detection.py           # Standalone pose detection using MediaPipe
├── requirements.txt            # Project dependencies
├── yolo11n-pose.pt             # YOLOv11 pose estimation model weights
├── README.md                   # Project documentation
└── video/                      # Directory for input video files
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd TC04
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Verify Model Weights:**
    Ensure `yolo11n-pose.pt` is present in the root directory. If not, it will be automatically downloaded by the Ultralytics library upon first run.

## Usage

### 1. Combined Analysis (Recommended)
Run the main pipeline to perform all detections (Pose, Emotion, Activity, Anomalies) on a video file.

```bash
python combined_detection.py
```
*Note: Ensure your input video is placed in the `video/` folder and the path is correctly set in the script.*

### 2. Emotion Detection Only
To analyze emotions in a specific video file:

```bash
python detect_expression_video.py
```

### 3. Real-Time Face Detection
To start a webcam feed for simple face detection:

```bash
python facil_detection.py
```

### 4. Pose Detection (MediaPipe)
To run the MediaPipe-based pose estimation:

```bash
python pose_detection.py
```

## Features

### Implemented
-   ✅ **Multi-Model Integration**: Combines YOLOv11 and DeepFace for holistic analysis.
-   ✅ **Activity Recognition**: Heuristic-based classification of user states (Standing, Sitting, Lying).
-   ✅ **Emotion Analysis**: Frame-by-frame emotion detection with confidence scores.
-   ✅ **Anomaly Detection**: Velocity-based monitoring for sudden movements.
-   ✅ **Scene Change Detection**: Identifies cuts in video footage.
-   ✅ **Reporting**: JSON export of analysis results.

## Requirements

-   Python 3.11+
-   OpenCV
-   Ultralytics (YOLOv11)
-   DeepFace
-   TensorFlow / Keras
-   MediaPipe

See `requirements.txt` for the complete list of dependencies and versions.

## License

This project is part of the FIAP Tech Challenge Phase 4.

## Author

Renato Penna

