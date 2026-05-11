import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import time
import urllib.request
from pathlib import Path

# --- Configuration ---
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

class HandTracker:
    def __init__(self, max_hands=2, detection_con=0.5):
        self._ensure_model_exists()
        self.results = None
        self.frame = None
        self.lock = threading.Lock()
        
        base = python.BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=python.BaseOptions.Delegate.CPU # CPU is more compatible across Linux setups
        )
        
        def callback(result, output_image, timestamp_ms):
            with self.lock:
                self.results = result

        options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            result_callback=callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        if not MODEL_PATH.exists():
            print(f"Downloading model to {MODEL_PATH}...")
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.detector.detect_async(mp_img, int(time.time() * 1000))

    def get_hands(self, img_width, img_height):
        hands_list = []
        with self.lock:
            if self.results and self.results.hand_landmarks:
                for i, landmarks in enumerate(self.results.hand_landmarks):
                    # Convert landmarks to pixel coordinates
                    lmList = []
                    for lm in landmarks:
                        lmList.append([int(lm.x * img_width), int(lm.y * img_height), lm.z])
                    
                    # Estimate bounding box
                    x_coords = [lm[0] for lm in lmList]
                    y_coords = [lm[1] for lm in lmList]
                    bbox = [min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)]
                    center = [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2]
                    
                    hands_list.append({
                        "lmList": lmList,
                        "bbox": bbox,
                        "center": center,
                        "raw_landmarks": landmarks
                    })
        return hands_list

    def draw_landmarks(self, img, hand):
        """Draws landmarks and connections on the image."""
        lmList = hand["lmList"]
        # Draw connections (simplified)
        connections = [
            (0,1), (1,2), (2,3), (3,4), # Thumb
            (0,5), (5,6), (6,7), (7,8), # Index
            (0,9), (9,10), (10,11), (11,12), # Middle
            (0,13), (13,14), (14,15), (15,16), # Ring
            (0,17), (17,18), (18,19), (19,20), # Pinky
            (5,9), (9,13), (13,17) # Palm
        ]
        for start, end in connections:
            cv2.line(img, tuple(lmList[start][:2]), tuple(lmList[end][:2]), (0, 255, 0), 2)
        for lm in lmList:
            cv2.circle(img, tuple(lm[:2]), 5, (0, 0, 255), cv2.FILLED)

    def fingers_up(self, hand):
        """Replicates cvzone's fingersUp logic."""
        fingers = []
        lmList = hand["lmList"]
        tipIds = [4, 8, 12, 16, 20]

        # Thumb
        # Check if thumb tip is to the left/right of its base (depends on hand side, but we simplify)
        if lmList[tipIds[0]][0] > lmList[tipIds[0] - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][1] < lmList[tipIds[id] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def get_camera():
    for i in [0, 1, 2]:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Opened camera index {i}")
            return cap
    return None
