import cv2
import numpy as np

from hand_tracking2 import FINGERTIPS


# ==========================================
# Shared rendering primitive
# ==========================================

def paint_filter_in_polygon(frame, polygon_pts, filtro_func):
    h, w = frame.shape[:2]

    x, y, bw, bh = cv2.boundingRect(polygon_pts)
    x, y = max(x, 0), max(y, 0)
    bw = min(bw, w - x)
    bh = min(bh, h - y)
    if bw <= 1 or bh <= 1:
        return frame

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts], 255)
    mask_roi = mask[y : y + bh, x : x + bw]

    roi = frame[y : y + bh, x : x + bw]
    filtered_roi = filtro_func(roi)

    mask_3ch = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    blended = (filtered_roi * mask_3ch + roi * (1.0 - mask_3ch)).astype(np.uint8)
    frame[y : y + bh, x : x + bw] = blended
    return frame


class ClosingGestureDetector:

    def __init__(self, close_ratio=0.16, open_ratio=0.30):
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.is_closed = False

    def update(self, width, frame_w):
        close_threshold = self.close_ratio * frame_w
        open_threshold = self.open_ratio * frame_w

        triggered = False
        if not self.is_closed and width < close_threshold:
            self.is_closed = True
            triggered = True
        elif self.is_closed and width > open_threshold:
            self.is_closed = False

        return triggered


# ==========================================
# Original two-point (index+thumb per hand) fixed-quad portal.
# Kept as-is in case you still want the simpler version anywhere.
# ==========================================

def portal_width(p1, p2, p3, p4):
    top_w = np.hypot(p3[0] - p1[0], p3[1] - p1[1])
    bottom_w = np.hypot(p4[0] - p2[0], p4[1] - p2[1])
    return (top_w + bottom_w) / 2.0


def render_portal(frame, p1, p2, p3, p4, filtro_func):
    full_polygon = np.array([p1, p3, p4, p2], dtype=np.int32)
    paint_filter_in_polygon(frame, full_polygon, filtro_func)
    cv2.polylines(frame, [full_polygon], isClosed=True, color=(255, 255, 255), thickness=1)
    return frame


# ==========================================
# NEW: flexible portal from all 10 fingertips (5 per hand)
# ==========================================

def fingertip_points(hand_landmarks, w, h):
    """Pixel coordinates of all 5 fingertips for one detected hand,
    in the fixed order defined by FINGERTIPS (thumb, index, middle, ring, pinky)."""
    return [(hand_landmarks[i].x * w, hand_landmarks[i].y * h) for i in FINGERTIPS]


def hands_spread(left_points, right_points):
    """Distance between the two hands' fingertip centroids. Plays the same
    role portal_width used to play: bring hands together to close the
    portal, spread them apart to reopen it — just generalized from 2 points
    per hand to the mean of all 5."""
    lx = sum(p[0] for p in left_points) / len(left_points)
    ly = sum(p[1] for p in left_points) / len(left_points)
    rx = sum(p[0] for p in right_points) / len(right_points)
    ry = sum(p[1] for p in right_points) / len(right_points)
    return np.hypot(rx - lx, ry - ly)


def flexible_portal_polygon(left_points, right_points):
    """Convex hull over all 10 fingertips. This is what lets the portal
    deform into whatever shape your two hands are making (splay your
    fingers wide = a wider hull; curl them in = a tighter one) instead of
    always being a straight-edged quad."""
    all_points = np.array(list(left_points) + list(right_points), dtype=np.float32)
    hull = cv2.convexHull(all_points)
    return hull.astype(np.int32)


class PointSmoother:
    """Exponential moving average per tracked point, keyed by name (e.g.
    'left_8' for the left hand's index tip). Landmark detection is noisy
    frame-to-frame, and a wobbly hand can be forgiven — a wobbly *hull edge*
    reads as jitter, so this settles it down. Call reset() when a hand
    leaves frame so it doesn't snap in from a stale position later."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self._smoothed = {}

    def smooth(self, key, point):
        prev = self._smoothed.get(key)
        if prev is None:
            self._smoothed[key] = point
        else:
            a = self.alpha
            self._smoothed[key] = (
                a * point[0] + (1 - a) * prev[0],
                a * point[1] + (1 - a) * prev[1],
            )
        return self._smoothed[key]

    def reset(self, prefix):
        for k in [k for k in self._smoothed if k.startswith(prefix)]:
            del self._smoothed[k]


def render_flexible_portal(frame, left_points, right_points, filtro_func, show_fingertips=True):
    polygon = flexible_portal_polygon(left_points, right_points)
    paint_filter_in_polygon(frame, polygon, filtro_func)
    cv2.polylines(frame, [polygon], isClosed=True, color=(255, 255, 255), thickness=1)

    if show_fingertips:
        for px, py in list(left_points) + list(right_points):
            cv2.circle(frame, (int(px), int(py)), 3, (255, 255, 255), -1)

    return frame