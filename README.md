# Filters

An augmented reality project that generates an interactive portal within the camera feed using real-time hand tracking. Through a natural gesture, users can switch between 8 different visual filters, which are rendered live inside the portal.

---

## Features

- Real-time hand tracking using MediaPipe Hands.
- Perspective-based portal dynamically constructed using the index finger and thumb tips of both hands.
- Eight visual filters applied exclusively within the portal area.
- Gesture-based filter switching: bringing the hands closer together triggers a transition to the next filter in the sequence.
- Hysteresis system to prevent accidental switching caused by hand tremors or tracking inaccuracies.

## Included Filters

| Filter | Description |
|--------|-------------|
| `filtro_grid` | Grid overlay on the original image |
| `filtro_1` | Threshold-based duotone effect using brightness levels |
| `filtro_2` | Black-and-white halftone dot pattern |
| `filtro_3` | Chromatic aberration with RGB channel separation |
| `filtro_5` | Thermal camera simulation using a colormap |
| `filtro_6` | Vintage sepia effect with vignetting and grain |
| `filtro_blanco` | Frosted glass effect applied to the image |
| `filtro_rosa` | Pink-magenta duotone halftone effect |

## Installation

Clone the repository:

```bash
git clone https://github.com/mishu006/Filters.git
cd Filters
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate       # Windows
source venv/bin/activate    # macOS / Linux
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

With the camera running, raise both hands with your index fingers and thumbs extended. The portal will automatically appear between them.

Bring your hands closer together to "close" the portal and advance to the next filter in the list.

Press **`q`** while the application window is