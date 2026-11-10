# Multi-Hand Detection with MediaPipe

Real-time webcam hand tracking that detects up to 6 hands at once, draws the 21 landmarks and skeleton for each hand, and shows a live hand count. Built with Python, OpenCV, and MediaPipe's Hand Landmarker (Tasks API).

## Features
- Detects multiple hands simultaneously (configurable via `MAX_HANDS`)
- Draws 21 landmarks and connections per hand
- Labels each hand and shows a live count
- Mirrored webcam view for natural interaction

## Requirements
- Python 3.9–3.12
- A working webcam

## Installation

```bash
git clone https://github.com/<your-username>/multi-hand-detection.git
cd multi-hand-detection

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Download the model
Download `hand_landmarker.task` from the official MediaPipe Hand Landmarker page and place it in the `models/` folder:

https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models

## Usage

```bash
python main.py
```

Press **ESC** to quit.

## Configuration

Edit the constants at the top of `main.py`:

| Variable | Default | Description |
|---|---|---|
| `MAX_HANDS` | 6 | Maximum number of hands to detect |
| `CAMERA_ID` | 0 | Webcam index (try 1 if you have multiple cameras) |
| `min_*_confidence` | 0.4 | Detection/tracking thresholds |

## How it works
1. Each frame is captured with OpenCV, mirrored, and converted BGR → RGB.
2. MediaPipe's `HandLandmarker` runs in `VIDEO` mode and returns 21 normalized landmarks per hand.
3. Landmarks are converted to pixel coordinates and drawn with OpenCV.

## License
MIT
