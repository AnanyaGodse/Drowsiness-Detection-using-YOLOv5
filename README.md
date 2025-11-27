# Drowsiness Detection System

A real-time drowsiness detection system using YOLOv5 to monitor driver alertness by detecting eye closure and yawning patterns.

## Overview

This project uses computer vision to detect signs of drowsiness in real-time video streams. The system identifies three key states: eyes closed, eyes open, and yawning. It processes these detections to determine alertness levels and triggers warnings when drowsiness is detected.

## Dataset

The dataset was created from scratch by recording video clips and extracting frames for annotation.

**Original Dataset:**
- Training: 169 images
- Validation: 38 images

**After Augmentation:**
- Training: 845 images
- Validation: 38 images

All images were annotated in YOLO format with three classes:
- Eyes Closed
- Eyes Open
- Yawning

## Model Performance

The model was fine-tuned using YOLOv5n and achieved strong results on the validation set:

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|--------|-----------|-----------|--------|---------|--------------|
| All | 38 | 38 | 0.970 | 0.954 | 0.992 | 0.772 |
| Eyes Closed | 13 | 13 | 0.915 | 1.000 | 0.990 | 0.732 |
| Eyes Open | 14 | 14 | 1.000 | 0.863 | 0.990 | 0.801 |
| Yawning | 11 | 11 | 0.995 | 1.000 | 0.995 | 0.784 |

**Model Details:**
- Architecture: YOLOv5n (fused)
- Parameters: 2,503,529
- GFLOPs: 7.1

## How It Works

The system uses a frame-based detection approach:

1. **Detection**: YOLOv5 identifies eyes closed, eyes open, and yawning states in each frame
2. **Tracking**: Maintains counters for consecutive frames of each state
3. **Alert Logic**: 
   - Critical alert: Eyes closed for more than 0.7 seconds (21 frames at 30fps)
   - Warning alert: Yawning detected for more than 0.5 seconds (15 frames at 30fps)
4. **Cooldown**: Prevents alert spam with a 3-second cooldown after critical alerts

## Inference Performance

The system processes video at an average of **137 FPS**, making it suitable for real-time applications.

**Test Video Results:**
- Frames processed: 617
- Average FPS: 137.17
- Critical alerts: 7
- Warning alerts: 0

## Features

- Real-time drowsiness detection with visual alerts
- Frame skipping optimization for faster processing
- On-screen statistics display showing:
  - Current eye closure duration
  - Yawn duration
  - Total alerts count
  - Real-time FPS
- Color-coded detection boxes
- Alert cooldown system to prevent excessive warnings

## Installation

```bash
pip install ultralytics opencv-python numpy
```

## Usage

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('path/to/best.pt')

# Run inference on video
results = test_on_video(
    model_path='path/to/best.pt',
    video_path='path/to/video.mp4',
    output_path='output.mp4',
    skip_frames=2
)
```

## Training Results

### Performance Curves

![F1 Curve](runs/detect/train/F1_curve.png)
![Precision Curve](runs/detect/train/P_curve.png)
![Recall Curve](runs/detect/train/R_curve.png)
![Confusion Matrix](runs/detect/train/confusion_matrix.png)

## Training Details

The model was trained with:
- Confidence threshold: 0.4
- IoU threshold: 0.5
- Maximum detections per frame: 3
- Data augmentation applied to training set

## Future Improvements

- Add head pose estimation for better drowsiness detection
- Implement audio alerts
- Add support for multiple face detection
- Deploy as a lightweight mobile application
- Integrate with vehicle control systems

## License

MIT License

## Acknowledgments

- YOLOv5 by Ultralytics
- Dataset collected and annotated manually
