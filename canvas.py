import time

import cv2
import numpy as np

# Palette cycled through with the fist gesture (BGR)
COLORS = [
    (60, 60, 255),    # red
    (60, 220, 60),    # green
    (255, 120, 60),   # blue
    (0, 220, 255),    # yellow
    (255, 60, 220),   # magenta
    (255, 255, 255),  # white
]


class AirCanvas:
    """Persistent drawing layer that floats above the camera feed.

    Strokes are drawn into a color layer + an alpha mask, so the drawing can
    be composited onto any live frame without carrying a black background.
    Supports multiple independent 'pens' at once (keyed by a label, e.g. a
    hand's "Left"/"Right" tag) so lifting one hand doesn't cut off a stroke
    being drawn with the other.
    """

    def __init__(self, width, height, thickness=6):
        self.width = width
        self.height = height
        self.thickness = thickness

        self.layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.mask = np.zeros((height, width), dtype=np.uint8)

        self.color_index = 0
        self._strokes = []      # committed strokes: list[list[(x, y)]]
        self._pens = {}         # label -> {"last": (x,y) | None, "stroke": [...]}

    @property
    def color(self):
        return COLORS[self.color_index % len(COLORS)]

    def next_color(self):
        self.color_index = (self.color_index + 1) % len(COLORS)

    @staticmethod
    def _px(point):
        return int(point[0]), int(point[1])

    def draw_to(self, pen_label, point):
        """Extend (or start) the stroke for this pen toward `point`."""
        pen = self._pens.setdefault(pen_label, {"last": None, "stroke": []})
        if pen["last"] is not None:
            cv2.line(self.layer, self._px(pen["last"]), self._px(point),
                      self.color, self.thickness, cv2.LINE_AA)
            cv2.line(self.mask, self._px(pen["last"]), self._px(point),
                      255, self.thickness, cv2.LINE_AA)
        pen["stroke"].append(point)
        pen["last"] = point

    def lift_pen(self, pen_label):
        """Pen-up for this pen: commit the in-progress stroke (if any)."""
        pen = self._pens.get(pen_label)
        if not pen:
            return
        if len(pen["stroke"]) > 1:
            self._strokes.append(pen["stroke"])
        pen["last"] = None
        pen["stroke"] = []

    def clear(self):
        self.layer[:] = 0
        self.mask[:] = 0
        self._strokes.clear()
        for pen in self._pens.values():
            pen["last"] = None
            pen["stroke"] = []

    def undo(self):
        """Remove the most recently completed stroke and redraw everything
        else from scratch. Strokes still in progress aren't affected."""
        if not self._strokes:
            return
        self._strokes.pop()
        self.layer[:] = 0
        self.mask[:] = 0

        remaining = self._strokes
        self._strokes = []
        for stroke in remaining:
            label = "__undo_redraw__"
            for point in stroke:
                self.draw_to(label, point)
            self.lift_pen(label)
        self._pens.pop("__undo_redraw__", None)

    def composite(self, frame):
        mask_3ch = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        blended = (
            self.layer.astype(np.float32) * mask_3ch
            + frame.astype(np.float32) * (1.0 - mask_3ch)
        )
        return blended.astype(np.uint8)

    def save(self, path=None):
        """Save just the drawing on a white background, like a page."""
        path = path or f"air_canvas_{int(time.time())}.png"
        white_bg = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        mask_3ch = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        out = (
            self.layer.astype(np.float32) * mask_3ch
            + white_bg.astype(np.float32) * (1.0 - mask_3ch)
        ).astype(np.uint8)
        cv2.imwrite(path, out)
        return path


def draw_hud(frame, canvas, mode_labels):
    """Small on-screen indicator: current color swatch + per-hand mode text."""
    out = frame.copy()

    # Color swatch, top-left
    cv2.rectangle(out, (10, 10), (40, 40), canvas.color, -1)
    cv2.rectangle(out, (10, 10), (40, 40), (255, 255, 255), 1)

    y = 60
    for label, mode in mode_labels.items():
        cv2.putText(out, f"{label}: {mode}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22

    return out