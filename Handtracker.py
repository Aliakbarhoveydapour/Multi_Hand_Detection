
import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

MAX_HANDS = 6
CAMERA_ID = 0

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find hand_landmarker.task at: {MODEL_PATH}"
    )

# ==========================================
# 2. MEDIAPIPE INITIALIZATION
# ==========================================

BaseOptions = mp.tasks.BaseOptions
Vision = mp.tasks.vision

base_options = BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = Vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=Vision.RunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4
)

# ==========================================
# 3. HAND CONNECTIONS
# ==========================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# ==========================================
# 4. DRAW HAND LANDMARKS
# ==========================================

def draw_hands(frame, results):

    height, width, _ = frame.shape

    number_of_hands = len(results.hand_landmarks)

    for hand_number, landmarks in enumerate(
        results.hand_landmarks
    ):

        points = []

        # Convert normalized coordinates to pixels
        for landmark in landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

        # Draw hand connections
        for start, end in HAND_CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )

        # Draw 21 landmarks
        for point_number, point in enumerate(points):

            cv2.circle(
                frame,
                point,
                5,
                (0, 0, 255),
                -1
            )

        # Label each detected hand
        wrist_x, wrist_y = points[0]

        cv2.putText(
            frame,
            f"Hand {hand_number + 1}",
            (wrist_x, max(25, wrist_y - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    # Display hand count
    cv2.putText(
        frame,
        f"Hands Detected: {number_of_hands}/{MAX_HANDS}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    return frame


# ==========================================
# 5. START WEBCAM
# ==========================================

cap = cv2.VideoCapture(CAMERA_ID)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Cannot access webcam!")

# ==========================================
# 6. REAL-TIME HAND DETECTION
# ==========================================

try:

    with Vision.HandLandmarker.create_from_options(
        options
    ) as detector:

        print("MediaPipe initialized successfully!")
        print(f"Maximum hands: {MAX_HANDS}")
        print("Press ESC to exit.")

        while True:

            success, frame = cap.read()

            if not success:
                print("Cannot read webcam frame.")
                break

            # Mirror camera
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Generate timestamp
            timestamp_ms = time.monotonic_ns() // 1_000_000

            # Detect hands
            results = detector.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # Draw results
            frame = draw_hands(frame, results)

            # Display webcam
            cv2.imshow(
                "MediaPipe - Four Hand Detection",
                frame
            )

            # Exit with ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:

    cap.release()
    cv2.destroyAllWindows()

    print("Program terminated.")
