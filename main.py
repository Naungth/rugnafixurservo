import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerResult
import cv2

model_path = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode



def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if result.hand_landmarks:
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            landmark_9 = hand_landmarks[9]
            hand_type = "Unknown"
            if result.handedness and i < len(result.handedness):
                hand_type = result.handedness[i][0].category_name
            print(f"Hand {i+1} ({hand_type}) - Landmark 9: x={landmark_9.x:.3f}, y={landmark_9.y:.3f}, z={landmark_9.z:.3f}")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,  # Detect up to 2 hands
    result_callback=result_callback,
)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        bgr_frame = image.copy()
        
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33  
        cv2.imshow("Camera", bgr_frame)
    cap.release()
    cv2.destroyAllWindows()