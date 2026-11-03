"""
shatter_portal.py

Turns the flexible fingertip portal into a "shattered glass" / kaleidoscope
mosaic like the reference screenshot: the hull polygon gets triangulated
into facets, each facet gets painted with its own distinct psychedelic
style (kaleidoscope mirror, swirl warp, polka dots, duotone), outlined in
white, plus an optional scanning-UI overlay (corner brackets + a coordinate
readout) for the "AR HUD" look.

Drop-in replacement for render_flexible_portal in main.py:

    from shatter_portal import render_shattered_portal
    ...
    frame = render_shattered_portal(frame, left_points, right_points)

It only needs the same left_points / right_points (10 fingertip pixel
coords) you're already computing in main.py — it doesn't touch FILTROS,
hand_tracking2.py, or geometry2.py at all.
"""

import random

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Shard styles — each takes a roi (BGR uint8) and returns a same-shape roi.
# Intentionally exaggerated/psychedelic, not "realistic" filters.
# ---------------------------------------------------------------------------

def _kaleidoscope(roi):
    h, w = roi.shape[:2]
    half_h, half_w = max(1, h // 2), max(1, w // 2)
    quad = cv2.resize(roi, (half_w, half_h))
    top = np.hstack([quad, cv2.flip(quad, 1)])
    bottom = cv2.flip(top, 0)
    mosaic = np.vstack([top, bottom])
    return cv2.resize(mosaic, (w, h))


def _swirl(roi, strength=3.5):
    h, w = roi.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    dx, dy = x_idx - cx, y_idx - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx) + strength * (r / (max(w, h) / 2.0 + 1e-5))
    map_x = (cx + r * np.cos(theta)).astype(np.float32)
    map_y = (cy + r * np.sin(theta)).astype(np.float32)
    return cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _polka_dots(roi):
    h, w = roi.shape[:2]
    base_color = cv2.mean(roi)[:3]
    canvas = np.full((h, w, 3), base_color, dtype=np.uint8)
    step = max(6, min(h, w) // 8)
    palette = [(200, 120, 255), (255, 200, 90), (120, 255, 210)]
    for yy in range(0, h, step):
        for xx in range(0, w, step):
            dot_color = palette[(xx // step + yy // step) % 3]
            cv2.circle(canvas, (xx, yy), max(1, step // 4), dot_color, -1)
    return canvas


def _duotone(roi, color_a=(255, 60, 120), color_b=(40, 20, 90)):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    a = np.array(color_a, dtype=np.float32)
    b = np.array(color_b, dtype=np.float32)
    out = gray[..., None] * a + (1 - gray[..., None]) * b
    return np.clip(out, 0, 255).astype(np.uint8)


SHARD_STYLES = [_kaleidoscope, _swirl, _polka_dots, _duotone]


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

def _triangulate(hull_points, frame_w, frame_h, interior_points=2):
    """Delaunay-triangulate the hull points, plus a couple of random
    interior 'steiner' points so shards come out uneven in size (like the
    reference's mix of big and small facets) instead of a plain fan."""
    pts = [tuple(p) for p in hull_points]
    poly_i32 = np.array(hull_points, dtype=np.int32)

    if interior_points > 0:
        x0, y0, w0, h0 = cv2.boundingRect(poly_i32)
        added, attempts = 0, 0
        while added < interior_points and attempts < 30:
            attempts += 1
            cand = (random.uniform(x0, x0 + w0), random.uniform(y0, y0 + h0))
            if cv2.pointPolygonTest(poly_i32, cand, False) >= 0:
                pts.append(cand)
                added += 1

    # Pad the rect so hull points that sit exactly on frame edges don't
    # get rejected by Subdiv2D for being "outside" the boundary.
    rect = (-10, -10, frame_w + 20, frame_h + 20)
    subdiv = cv2.Subdiv2D(rect)
    for p in pts:
        subdiv.insert((float(p[0]), float(p[1])))

    triangles = []
    for t in subdiv.getTriangleList():
        tri = np.array([[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]], dtype=np.float32)
        centroid = tri.mean(axis=0)
        if cv2.pointPolygonTest(poly_i32, tuple(centroid), False) >= 0:
            triangles.append(tri.astype(np.int32))
    return triangles


# ---------------------------------------------------------------------------
# Painting
# ---------------------------------------------------------------------------

def _paint_triangle(frame, tri_pts, style_func):
    h, w = frame.shape[:2]
    x, y, bw, bh = cv2.boundingRect(tri_pts)
    x, y = max(x, 0), max(y, 0)
    bw = min(bw, w - x)
    bh = min(bh, h - y)
    if bw <= 1 or bh <= 1:
        return

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, tri_pts, 255)
    mask_roi = mask[y:y + bh, x:x + bw]
    if mask_roi.max() == 0:
        return

    roi = frame[y:y + bh, x:x + bw]
    styled = style_func(roi)

    mask_3ch = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    blended = (styled * mask_3ch + roi * (1.0 - mask_3ch)).astype(np.uint8)
    frame[y:y + bh, x:x + bw] = blended


def _draw_hud(frame, hull_points, right_points):
    """Corner brackets + a coordinate readout, like a scanning/targeting UI."""
    color = (255, 255, 255)
    bracket_len = 14

    xs = [p[0] for p in hull_points]
    ys = [p[1] for p in hull_points]
    corners = [(min(xs), min(ys)), (max(xs), min(ys)), (min(xs), max(ys)), (max(xs), max(ys))]
    for cx, cy in corners:
        cx, cy = int(cx), int(cy)
        cv2.line(frame, (cx, cy), (cx + bracket_len, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + bracket_len), color, 1, cv2.LINE_AA)

    if right_points:
        rx, ry = right_points[1]  # index tip
        label = f"x:{int(rx)} y:{int(ry)}"
        cv2.putText(frame, label, (int(rx) + 10, int(ry)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (int(rx) + 10, int(ry)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)


_frame_counter = 0


def render_shattered_portal(frame, left_points, right_points, show_hud=True, style_funcs=None):
    """Drop-in replacement for render_flexible_portal: same hull, but
    triangulated into shards, each painted with a different psychedelic
    style, outlined, with an optional AR-HUD overlay."""
    global _frame_counter
    _frame_counter += 1

    h, w = frame.shape[:2]
    all_points = list(left_points) + list(right_points)
    hull = cv2.convexHull(np.array(all_points, dtype=np.float32))
    hull_points = [tuple(p[0]) for p in hull]

    styles = style_funcs or SHARD_STYLES
    triangles = _triangulate(hull_points, w, h, interior_points=2)

    # Sort left-to-right so style assignment stays roughly stable frame to
    # frame even though the exact triangulation reshuffles as the hand moves.
    triangles.sort(key=lambda t: t[:, 0].mean())

    # Slowly rotate which style lands on which shard, so the mosaic evolves
    # over time instead of being frozen or flickering every frame.
    style_offset = _frame_counter // 90

    for i, tri in enumerate(triangles):
        style = styles[(i + style_offset) % len(styles)]
        _paint_triangle(frame, tri, style)
        cv2.polylines(frame, [tri], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    hull_int = hull.astype(np.int32)
    cv2.polylines(frame, [hull_int], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    if show_hud:
        _draw_hud(frame, hull_points, right_points)

    return frame