"""
Gesture-controlled Brave launcher
----------------------------------
Uses your webcam + MediaPipe hand tracking to control Brave with gestures:

    ✋ 5 fingers          -> open/focus Brave
    👍 thumb only         -> open a new tab
    ☝️  index finger only  -> open Reddit
    ✌️  index + middle     -> open ChatGPT
    👋 wave your hand      -> quit this script (does not close Brave)

Install dependencies:
    pip install opencv-python mediapipe
    pip install pygetwindow      # Windows only — lets the script bring Brave to front

After opening/switching, the script also tries to raise Brave above other
windows automatically (see focus_brave()) so you don't need to alt-tab to
it. On Linux this needs wmctrl or xdotool installed (e.g. `sudo apt install
wmctrl`) — without one of those it'll still open Brave, just without
auto-focusing it.

NOTE on mediapipe versions: the classic `mp.solutions.hands` API used here
was removed in some newer mediapipe releases (0.10.3x+). If `import
mediapipe` works but you get an AttributeError on `mp.solutions`, install
an older release that still has it:
    pip install "mediapipe==0.10.14"

Edit BRAVE_WIN_PATHS / BRAVE_LINUX_EXES below if Brave isn't found
automatically on your machine.

Press 'q' in the video window to quit manually.
"""

import math
import os
import shutil
import platform
import subprocess
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# ---------- Tunables ----------
CAMERA_INDEX = 0
GESTURE_HOLD_SEC = 0.7        # gesture must be held this long (real time, not frame count) to confirm
ACTION_COOLDOWN_SEC = 1.5     # min gap between two triggered actions
FINGER_EXTEND_RATIO = 1.15    # how much farther a tip must be than its base joint to count as "extended"
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.7
WAVE_TIME_WINDOW = 1.2        # only look at the last N seconds of movement for a wave
WAVE_MIN_SAMPLES = 8          # need at least this many samples inside the time window
WAVE_MIN_STEP = 0.02          # ignore frame-to-frame jitter smaller than this (fraction of frame width)
WAVE_MIN_RANGE = 0.22         # min horizontal travel (fraction of frame width) to count as a wave
WAVE_MIN_DIRECTION_CHANGES = 4
WAVE_MIN_FINGERS_UP = 3       # a wave must look like a fairly open hand, not a specific finger count
WAVE_HISTORY_MAXLEN = 60      # raw sample buffer; filtered down to WAVE_TIME_WINDOW internally

BRAVE_WIN_PATHS = [
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
]
BRAVE_LINUX_EXES = ["brave-browser", "brave", "brave-browser-stable"]
# -------------------------------

GESTURES = {
    (1, 1, 1, 1, 1): ("Open Brave", None),
    (1, 0, 0, 0, 0): ("New tab", "about:blank"),
    (0, 1, 0, 0, 0): ("Reddit", "https://www.reddit.com"),
    (0, 1, 1, 0, 0): ("ChatGPT", "https://chatgpt.com"),
}


def build_brave_command(url=None):
    """Return the subprocess argument list to launch/focus Brave, or None if not found."""
    system = platform.system()

    if system == "Windows":
        for path in BRAVE_WIN_PATHS:
            if os.path.exists(path):
                args = [path]
                if url:
                    args.append(url)
                return args
        return None

    if system == "Darwin":  # macOS
        args = ["open", "-a", "Brave Browser"]
        if url:
            args += ["--args", url]
        return args

    # Linux
    for exe in BRAVE_LINUX_EXES:
        if shutil.which(exe):
            args = [exe]
            if url:
                args.append(url)
            return args
    return None


def focus_brave(retries=15, delay=0.2):
    """Try to raise Brave above other windows. Runs on its own thread (called
    from open_brave) so the retry/sleep loop never blocks the camera feed."""
    system = platform.system()
    try:
        if system == "Windows":
            import pygetwindow as gw
            for _ in range(retries):
                wins = [w for w in gw.getAllWindows() if "brave" in w.title.lower()]
                if wins:
                    win = wins[0]
                    if win.isMinimized:
                        win.restore()
                    win.activate()
                    return
                time.sleep(delay)

        elif system == "Darwin":  # macOS
            subprocess.run(
                ["osascript", "-e", 'tell application "Brave Browser" to activate'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        else:  # Linux — needs wmctrl or xdotool installed
            time.sleep(delay * 2)  # give the window a moment to appear
            for cmd in (["wmctrl", "-a", "Brave"],
                        ["xdotool", "search", "--name", "Brave", "windowactivate"]):
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
                    return
                except FileNotFoundError:
                    continue
    except Exception as e:
        print(f"(couldn't bring Brave to the front: {e})")


def open_brave(url=None):
    args = build_brave_command(url)
    if args is None:
        print("Could not find Brave on this system. Edit BRAVE_WIN_PATHS / "
              "BRAVE_LINUX_EXES at the top of the script with the correct path.")
        return
    try:
        subprocess.Popen(args)
    except Exception as e:
        print(f"Failed to launch Brave: {e}")
        return
    threading.Thread(target=focus_brave, daemon=True).start()


def fingers_up(landmarks, handedness_label):
    """Return [thumb, index, middle, ring, pinky] as 1 (extended) or 0 (folded).

    Uses relative distances from stable reference points rather than raw
    x/y comparisons. The old version compared thumb tip.x vs ip.x directly,
    which only works when the hand is upright and facing the camera — tilt
    your hand (scratching your head, making a fist at an angle) and it
    frequently misreads a folded thumb as extended. Distance-based checks
    hold up much better across hand rotations.
    """
    def dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    wrist = landmarks[0]
    pinky_mcp = landmarks[17]

    fingers = []

    # Thumb: extended if the tip sits much farther from the pinky-base than
    # the thumb's own base joint does (folded thumbs stay close to the palm).
    thumb_tip_dist = dist(landmarks[4], pinky_mcp)
    thumb_base_dist = dist(landmarks[2], pinky_mcp)
    fingers.append(1 if thumb_tip_dist > thumb_base_dist * FINGER_EXTEND_RATIO else 0)

    # Other four fingers: extended if the tip is meaningfully farther from
    # the wrist than its own pip joint is — works regardless of hand tilt.
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        tip_dist = dist(landmarks[tip], wrist)
        pip_dist = dist(landmarks[pip], wrist)
        fingers.append(1 if tip_dist > pip_dist * FINGER_EXTEND_RATIO else 0)

    return fingers


def is_waving(history):
    """history holds (timestamp, x) samples. Only the last WAVE_TIME_WINDOW
    seconds are considered, so a slow drift over many seconds (e.g. repositioning
    your hand between gestures) can't accumulate enough 'direction changes' to
    look like a wave the way a frame-count-only check could."""
    now = time.time()
    recent = [x for t, x in history if now - t <= WAVE_TIME_WINDOW]
    if len(recent) < WAVE_MIN_SAMPLES:
        return False

    diffs = np.diff(recent)
    # drop jitter-scale movements so tiny tracking noise isn't counted as a direction change
    diffs = diffs[np.abs(diffs) >= WAVE_MIN_STEP]
    if len(diffs) < 2:
        return False

    signs = np.sign(diffs)
    direction_changes = int(np.sum(signs[1:] != signs[:-1]))
    value_range = max(recent) - min(recent)
    return direction_changes >= WAVE_MIN_DIRECTION_CHANGES and value_range >= WAVE_MIN_RANGE


def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                            min_tracking_confidence=MIN_TRACKING_CONFIDENCE)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open webcam. Try a different CAMERA_INDEX.")
        return

    wrist_history = deque(maxlen=WAVE_HISTORY_MAXLEN)  # (timestamp, x) samples
    current_gesture = None
    gesture_start_time = None
    last_triggered_gesture = None
    last_action_time = 0.0
    status_text = "Show a gesture..."

    print("Gesture control running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture_this_frame = None

        if result.multi_hand_landmarks and result.multi_handedness:
            hand_landmarks = result.multi_hand_landmarks[0]
            handedness_label = result.multi_handedness[0].classification[0].label
            landmarks = hand_landmarks.landmark

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            pattern = fingers_up(landmarks, handedness_label)

            # Use the average of wrist + finger-base knuckles as the tracked point —
            # much steadier than the wrist landmark alone, which jitters more as
            # fingers move (this was the main source of false "wave" triggers).
            palm_x = np.mean([landmarks[i].x for i in (0, 5, 9, 13, 17)])
            wrist_history.append((time.time(), palm_x))

            # Only treat it as a wave candidate if the hand is fairly open —
            # keeps it from firing while you're mid-transition into a specific
            # finger-count gesture.
            if sum(pattern) >= WAVE_MIN_FINGERS_UP and is_waving(wrist_history):
                status_text = "Wave detected - quitting!"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                cv2.imshow("Gesture Control (press q to quit)", frame)
                cv2.waitKey(500)
                break

            pattern = tuple(pattern)
            if pattern in GESTURES:
                gesture_this_frame = pattern

        # debounce: require the gesture to be held continuously for GESTURE_HOLD_SEC
        # (real time, not a frame count — stays consistent regardless of processing speed)
        now = time.time()
        if gesture_this_frame is not None and gesture_this_frame == current_gesture:
            held_for = now - gesture_start_time
        else:
            current_gesture = gesture_this_frame
            gesture_start_time = now
            held_for = 0.0

        if gesture_this_frame is None:
            # hand lowered / unrecognized pose -> allow the same gesture to fire again next time
            last_triggered_gesture = None
            status_text = "Show a gesture..."
        elif gesture_this_frame != last_triggered_gesture:
            label, _ = GESTURES[gesture_this_frame]
            remaining = max(0.0, GESTURE_HOLD_SEC - held_for)
            status_text = f"{label}... hold ({remaining:.1f}s)" if remaining > 0 else f"{label}!"

        if (gesture_this_frame is not None
                and held_for >= GESTURE_HOLD_SEC
                and gesture_this_frame != last_triggered_gesture
                and now - last_action_time > ACTION_COOLDOWN_SEC):
            label, url = GESTURES[gesture_this_frame]
            status_text = f"{label}!"
            print(status_text)
            open_brave(url)
            last_triggered_gesture = gesture_this_frame
            last_action_time = now

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.imshow("Gesture Control (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()