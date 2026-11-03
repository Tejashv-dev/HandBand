"""
Sci-fi hologram filtro for the AR portal app.

Drop-in for FILTROS — same calling convention as your other filters
(`filtro_func(roi) -> filtered_roi`, a BGR numpy array in, same shape out).
The only difference is this one is a *callable object* instead of a bare
function, because a believable hologram needs to animate frame to frame
(scan lines drifting, flicker, the odd glitch band) — that needs a frame
counter to persist across calls, which a stateless function can't do.
It's still used exactly the same way: `hologram(roi)`.

Layers, back to front:
  1. Cyan/blue color grade      - crushes red, pushes blue/green -> that
                                   classic "blue light" hologram palette
  2. Ghost/parallax double-image - a faint vertically-offset duplicate,
                                   like a projected volume you're seeing
                                   from slightly the wrong angle -> reads
                                   as depth without needing real 3D data
  3. Perspective grid floor      - converging lines + growing horizontal
                                   spacing in the lower third, the classic
                                   "Tony Stark UI" grid -> this is what
                                   sells the 3D-ish look the most
  4. Edge rim-glow                - Canny edges, blurred and tinted cyan,
                                   added back like a Fresnel/rim light
  5. Chromatic aberration        - R/B channels shifted a couple px
                                   opposite directions -> cheap projector
                                   fringing
  6. Animated scan lines          - thin dark bands that drift downward
  7. Flicker                      - small random per-frame brightness jitter
  8. Occasional glitch band       - rare horizontal slice gets rolled
                                   sideways for one frame
  9. Grain                        - light noise so it doesn't look flat
 10. Vertical fade                - top/bottom soften, like light falloff
                                   at the edges of a projected volume
"""

import random

import cv2
import numpy as np


class HologramFilter:
    def __init__(self):
        self.frame_count = 0

    def __call__(self, roi):
        self.frame_count += 1
        h, w = roi.shape[:2]
        if h < 4 or w < 4:
            return roi

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 1. Cyan/blue color grade
        b, g, r = cv2.split(roi.astype(np.float32))
        b = np.clip(b * 1.25 + 20, 0, 255)
        g = np.clip(g * 1.05 + 10, 0, 255)
        r = np.clip(r * 0.35, 0, 255)
        img = cv2.merge([b, g, r])

        # 2. Ghost / parallax double-image
        offset = max(2, h // 60)
        ghost = np.zeros_like(img)
        ghost[offset:, :] = img[:-offset, :]
        img = cv2.addWeighted(img, 1.0, ghost, 0.35, 0)

        # 3. Perspective grid floor (bottom third)
        grid_h = h // 3
        if grid_h > 4:
            grid = np.zeros((grid_h, w, 3), dtype=np.float32)
            vanish_x = w / 2.0
            for i in range(-4, 5):
                x1 = vanish_x + i * w * 0.6
                cv2.line(grid, (int(vanish_x), 0), (int(x1), grid_h),
                         (255, 255, 140), 1, cv2.LINE_AA)
            for j in range(1, 6):
                yy = int(grid_h * (j / 6.0) ** 1.6)
                if yy < grid_h:
                    cv2.line(grid, (0, yy), (w, yy), (255, 255, 140), 1, cv2.LINE_AA)
            sub = img[h - grid_h:, :]
            img[h - grid_h:, :] = cv2.addWeighted(sub, 1.0, grid, 0.45, 0)

        # 4. Edge rim-glow
        edges = cv2.Canny(gray, 60, 160)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        glow = cv2.GaussianBlur(edges, (9, 9), 0).astype(np.float32) / 255.0
        cyan = np.zeros_like(img)
        cyan[..., 0] = 255
        cyan[..., 1] = 255
        cyan[..., 2] = 60
        img = img + cyan * glow[..., None] * 0.9

        # 5. Chromatic aberration
        shift = 2
        bb, gg, rr = cv2.split(img)
        bb = np.roll(bb, shift, axis=1)
        rr = np.roll(rr, -shift, axis=1)
        img = cv2.merge([bb, gg, rr])

        # 6. Animated scan lines
        scan_period = 6
        offset_row = self.frame_count % scan_period
        rows = np.arange(h)
        scan_mask = np.where((rows + offset_row) % scan_period < 2, 0.55, 1.0).astype(np.float32)
        img *= scan_mask[:, None, None]

        # 7. Flicker
        img *= 0.9 + 0.1 * random.random()

        # 8. Occasional glitch band
        if random.random() < 0.06:
            band_h = max(2, h // 12)
            y0 = random.randint(0, max(0, h - band_h))
            dx = random.choice([-8, -5, 5, 8])
            img[y0:y0 + band_h] = np.roll(img[y0:y0 + band_h], dx, axis=1)

        # 9. Grain
        img += np.random.normal(0, 6, img.shape).astype(np.float32)

        # 10. Vertical fade
        fade_len = max(1, h // 6)
        fade = np.ones(h, dtype=np.float32)
        fade[:fade_len] = np.linspace(0.5, 1.0, fade_len)
        fade[-fade_len:] = np.linspace(1.0, 0.5, fade_len)
        img *= fade[:, None, None]

        return np.clip(img, 0, 255).astype(np.uint8)


# A ready-to-use instance — import this into filters.py
hologram = HologramFilter()