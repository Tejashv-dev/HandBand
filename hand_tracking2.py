"""Landmark index constants for MediaPipe's 21-point hand model.

NOTE: if your existing hand_tracking.py has other helper functions in it,
keep those — just make sure these fingertip constants (and FINGERTIPS) are
present, since geometry.py now imports FINGERTIPS.
"""

# Fingertip landmark indices. Same indexing whether you're on the legacy
# mp.solutions.hands API or the newer mediapipe.tasks vision API used in
# main.py — it's the same underlying 21-point hand landmark model.
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# All five fingertips in a fixed order, used to build the flexible portal.
FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)