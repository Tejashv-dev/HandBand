import cv2
import numpy as np

from hologram import hologram



def filtro_1(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    out = np.zeros_like(roi)
    out[gray < 60] = (15, 8, 10)
    out[(gray >= 60) & (gray < 130)] = (118, 30, 214)
    out[(gray >= 130) & (gray < 195)] = (35, 140, 235)
    out[gray >= 195] = (235, 240, 240)
    return out


def filtro_2(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell = 6
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (xx % cell) - cell / 2
    cy = (yy % cell) - cell / 2
    dist_center = np.sqrt(cx**2 + cy**2)
    radius = (1 - gray / 255.0) * (cell / 1.4)
    dot_mask = dist_center < radius
    out = np.full_like(roi, 245)
    out[dot_mask] = (15, 15, 15)
    return out


def filtro_3(roi: np.ndarray) -> np.ndarray:
    shift = 6
    b, g, r = cv2.split(roi)
    r_shift = np.roll(r, -shift, axis=1)
    b_shift = np.roll(b, shift, axis=1)
    out = cv2.merge([b_shift, g, r_shift])
    out[::3, :, :] = (out[::3, :, :] * 0.72).astype(np.uint8)
    return out


def filtro_5(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def filtro_6(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    sepia_kernel = np.array(
        [
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189],
        ]
    )
    sepia = cv2.transform(roi, sepia_kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cy, cx = h / 2, w / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2) or 1.0
    vignette = np.clip(1 - 0.5 * (dist / max_dist), 0, 1)[..., None]

    out = (sepia * vignette).astype(np.uint8)
    noise = np.random.randint(0, 25, out.shape, dtype=np.uint8)
    out = cv2.add(out, noise)
    return out


def filtro_blanco(roi: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(roi, (35, 35), 0)
    white = np.full_like(roi, 255)
    out = cv2.addWeighted(blurred, 0.55, white, 0.45, 0)
    return out


def filtro_rosa(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell = 5
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (xx % cell) - cell / 2
    cy = (yy % cell) - cell / 2
    dist_center = np.sqrt(cx**2 + cy**2)
    radius = (1 - gray / 255.0) * (cell / 1.3)
    dot_mask = dist_center < radius

    out = np.full_like(roi, (215, 190, 245))
    out[dot_mask] = (55, 20, 130)
    return out


def filtro_grid(roi: np.ndarray) -> np.ndarray:
    out = roi.copy()
    h, w = out.shape[:2]
    step = 22
    color = (235, 235, 235)

    overlay = out.copy()
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), color, 1)

    out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)
    return out


# ==========================================
# NEW FILTERS
# ==========================================

def filtro_neon(roi: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(roi, 75, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    out = np.zeros_like(roi)
    # Color edges in neon pink/purple (BGR format)
    out[edges > 0] = (255, 50, 200)
    return out


def filtro_pixelado(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    pixel_size = 12
    # Prevent crashing if the portal gets too small
    if w < pixel_size or h < pixel_size:
        return roi
    
    # Downscale then upscale without interpolation to create blocky pixels
    small = cv2.resize(roi, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def filtro_glitch(roi: np.ndarray) -> np.ndarray:
    out = roi.copy()
    h, w = out.shape[:2]
    
    # Create random horizontal slices and shift them
    for _ in range(8):
        y1 = np.random.randint(0, max(1, h - 10))
        y2 = y1 + np.random.randint(5, 20)
        shift = np.random.randint(-20, 20)
        out[y1:y2] = np.roll(out[y1:y2], shift, axis=1)
    
    # Add VHS-style scanlines by darkening every 3rd row
    out[::3, :, :] = (out[::3, :, :] * 0.6).astype(np.uint8) 
    return out


def filtro_poster(roi: np.ndarray) -> np.ndarray:
    # Quantize colors into large buckets to create a comic/poster effect
    out = (roi // 64) * 64
    out = np.clip(out + 32, 0, 255).astype(np.uint8)
    return out


# ==========================================
# NEW FILTERS - ROUND 2 (the crazy ones)
# ==========================================

def filtro_kaleidoscopio(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    if h < 8 or w < 8:
        return roi

    center = (w / 2, h / 2)
    max_radius = np.sqrt(center[0] ** 2 + center[1] ** 2)

    # Unroll the image into polar coordinates: rows = angle, cols = radius
    polar = cv2.warpPolar(roi, (w, h), center, max_radius, cv2.WARP_POLAR_LINEAR)

    # Grab one narrow angular wedge, mirror it, and stack that mirrored pair
    # over and over until it fills the full 360 degrees -> repeating symmetry
    slices = 8
    wedge_h = max(1, h // slices)
    wedge = polar[:wedge_h]
    mirrored = cv2.flip(wedge, 0)
    pattern = np.concatenate([wedge, mirrored], axis=0)

    repeats = int(np.ceil(h / pattern.shape[0])) + 1
    tiled = np.tile(pattern, (repeats, 1, 1))[:h]

    # Roll the repeating wedge back into Cartesian space
    out = cv2.warpPolar(
        tiled, (w, h), center, max_radius,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    return out


def filtro_fuego(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape

    # Fire palette in BGR: black -> deep red -> orange -> yellow -> white
    positions = np.array([0.00, 0.25, 0.55, 0.80, 1.00])
    colors = np.array(
        [
            (5, 0, 0),
            (10, 15, 120),
            (10, 110, 220),
            (60, 200, 250),
            (230, 240, 255),
        ],
        dtype=np.float32,
    )

    out = np.empty((h, w, 3), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(gray, positions, colors[:, c])

    # Flicker: wiggle each row sideways with a sine wave for a "licking flame" feel
    phase = np.random.uniform(0, 2 * np.pi)
    rows = np.arange(h)
    shift = (5 * np.sin(rows / 8.0 + phase)).astype(np.int32)
    col_idx = (np.arange(w)[None, :] - shift[:, None]) % w
    row_idx = rows[:, None]
    out = out[row_idx, col_idx]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def filtro_remolino(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    cx, cy = w / 2, h / 2
    max_radius = np.sqrt(cx**2 + cy**2) or 1.0

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dx = xx - cx
    dy = yy - cy
    radius = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    # Twist grows toward the center -> classic vortex / whirlpool pull
    strength = 3.5
    twist = strength * np.clip(1 - radius / max_radius, 0, 1)
    new_angle = angle + twist

    map_x = (cx + radius * np.cos(new_angle)).astype(np.float32)
    map_y = (cy + radius * np.sin(new_angle)).astype(np.float32)

    out = cv2.remap(
        roi, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return out


def filtro_anaglifo3d(roi: np.ndarray) -> np.ndarray:
    shift = 8
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Fake a stereo pair by nudging the same image left/right, then
    # recombine as a red/cyan anaglyph like old-school 3D glasses
    right_eye = np.roll(gray, -shift, axis=1)
    left_eye = np.roll(gray, shift, axis=1)

    out = np.zeros_like(roi)
    out[..., 2] = right_eye  # Red channel
    out[..., 1] = left_eye   # Green channel
    out[..., 0] = left_eye   # Blue channel (green+blue = cyan)
    return out


def filtro_matrix(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    col_width = 3
    if w < col_width or h < 4:
        return roi

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Dim, green-tinted silhouette so the original shape stays recognizable
    base = np.zeros_like(roi)
    base[..., 1] = (gray * 0.35).astype(np.uint8)

    # Cascading green "code rain" columns, each with its own length/fade
    n_cols = w // col_width
    rain = np.zeros((h, w), dtype=np.uint8)
    for i in range(n_cols):
        x = i * col_width
        drop_len = np.random.randint(h // 4 + 1, h + 1)
        drop_start = np.random.randint(0, h)
        ys = (np.arange(drop_start, drop_start + drop_len) % h)
        fade = np.linspace(255, 40, drop_len).astype(np.uint8)
        rain[ys, x:x + col_width] = fade[:, None]

    out = base.copy()
    out[..., 1] = cv2.max(out[..., 1], rain)

    # Bright white-green leading edge on the hottest rain pixels for that glow
    bright_mask = rain > 220
    out[bright_mask] = (200, 255, 200)
    return out


# ==========================================
# FILTER SEQUENCE
# ==========================================

FILTROS = [
     filtro_matrix,
     filtro_pixelado,
    filtro_grid,
    filtro_1,
    filtro_2,
    filtro_3,
    filtro_5,
    filtro_6,
    filtro_blanco,
    filtro_rosa,
    filtro_neon,
    
    filtro_glitch,
    filtro_poster,

    filtro_fuego,          # NEW
    filtro_remolino,       # NEW
    filtro_anaglifo3d,     # NEW
            # NEW
    hologram,
]