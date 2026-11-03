import os
import urllib.request
import cv2
import mediapipe as mp
import pyvirtualcam
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hand_tracking import INDEX_TIP, THUMB_TIP
from geometry import render_portal, portal_width, ClosingGestureDetector
from filters import FILTROS

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def main():
    ensure_model()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check permissions and index.")

    # Match the virtual cam's resolution/fps to what the real camera gives us
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    filtro_index = 0
    closing_detector = ClosingGestureDetector()
    frame_timestamp_ms = 0

    # fmt=PixelFormat.BGR lets us send OpenCV frames directly, no channel swap needed
    with pyvirtualcam.Camera(width=w, height=h, fps=int(fps), fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        print(f"Virtual camera running: {cam.device}")
        print("Select this device as your camera in Zoom / Meet / Teams.")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            frame_timestamp_ms += 33
            results = detector.detect_for_video(mp_image, frame_timestamp_ms)

            left_hand = None
            right_hand = None

            if results.hand_landmarks and results.handedness:
                for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                    raw_label = handedness[0].category_name
                    label = "Right" if raw_label == "Left" else "Left"

                    if label == "Left":
                        left_hand = hand_landmarks
                    else:
                        right_hand = hand_landmarks

            if left_hand is not None and right_hand is not None:
                p1 = (left_hand[INDEX_TIP].x * fw, left_hand[INDEX_TIP].y * fh)
                p2 = (left_hand[THUMB_TIP].x * fw, left_hand[THUMB_TIP].y * fh)
                p3 = (right_hand[INDEX_TIP].x * fw, right_hand[INDEX_TIP].y * fh)
                p4 = (right_hand[THUMB_TIP].x * fw, right_hand[THUMB_TIP].y * fh)

                width = portal_width(p1, p2, p3, p4)

                if closing_detector.update(width, fw):
                    filtro_index = (filtro_index + 1) % len(FILTROS)

                frame = render_portal(frame, p1, p2, p3, p4, FILTROS[filtro_index])

            # Send to the virtual camera (this is what Zoom/Meet/Teams will see)
            cam.send(frame)
            cam.sleep_until_next_frame()

            # Optional local preview window; comment out if you don't want it
            cv2.imshow("AR Filters Portal (local preview)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()