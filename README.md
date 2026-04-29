# Nebulous YOLO Detector

<p align="center">
  <img src="https://github.com/user-attachments/assets/22526dea-7aef-4f1b-b477-4fbb6e9e6ce6" width="600">
</p>

A project for detecting objects in a Nox/NoxPlayer emulator window with Ultralytics YOLOv8. The program can save a screenshot with bounding boxes around detected objects and record a short video with real-time detection.

The model can be fine-tuned and extended for your own use cases.

You can download the model weights here:

https://drive.google.com/file/d/1a6VgKJIkP52Mtc_cmnrrTojxvobxQ0TT/view?usp=drivesdk

## Project Structure

- `nebulous_detector/` - the main package with shared logic.
- `main.py` - compatible entry point for one-off detection.
- `videomain.py` - compatible entry point for recording video with detection.
- `train_yolo.py` - CLI for starting YOLO training.
- `requirements.txt` - project dependencies.

Inside the package:

- `config.py` - paths, thresholds, classes, colors, FPS, and recording duration.
- `window_capture.py` - Nox window lookup and frame capture.
- `detection.py` - YOLO loading and detection result extraction.
- `drawing.py` - rendering bounding boxes and labels.
- `image_app.py` - one-off detection scenario.
- `video_app.py` - video recording scenario.
- `training.py` - model training function.

## Requirements

- Windows.
- Python 3.10+.
- Running Nox instance.
- YOLO weights file `best.pt` in the project root.
- For better performance, an NVIDIA GPU with a CUDA-compatible PyTorch build is recommended.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

After installation, place the model file `best.pt` in the project root.

## Quick Start

One-off detection:

```bash
python main.py
```

You can also run the module directly:

```bash
python -m nebulous_detector.image_app
```

After the run, a file like this will be created:

```text
detection_output_20260427_153000.png
```

Recording video with detection:

```bash
python videomain.py
```

You can also run the module directly:

```bash
python -m nebulous_detector.video_app
```

After the run, a file like this will be created:

```text
detection_video_20260427_153000.mp4
```

## Configuration

The main settings are in `nebulous_detector/config.py`:

```python
WEIGHTS_PATH = "best.pt"
CONF_THRESH = 0.25
FPS = 50
RECORD_SECONDS = 30
CLASS_NAMES = ["sphere", "plasma", "rainbow", "warning", "blob"]
SEARCH_WINDOW_KEYWORDS = ["nox", "noxplayer"]
```

If your Nox window has a different title, add the required word to `SEARCH_WINDOW_KEYWORDS`.

## Model Training

Create `data.yaml` in the Ultralytics YOLO format:

```yaml
train: path/to/train/images
val: path/to/val/images
nc: 5
names: ["sphere", "plasma", "rainbow", "warning", "blob"]
```

Start training:

```bash
python train_yolo.py --data data.yaml --epochs 50 --batch 16 --model yolov8n.pt
```

Training results will be saved to `runs/detect`.

## Troubleshooting

If the Nox window is not found, make sure the emulator is running and the window title contains `Nox` or `NoxPlayer`.

If the model does not load, check that `best.pt` is located in the project root. If needed, update `WEIGHTS_PATH` in `nebulous_detector/config.py`.

If video recording is slow, lower `FPS`, reduce the input `imgsz` in `nebulous_detector/video_app.py`, or use a GPU.
