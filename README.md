# Metro Exit Counter v2.0

An AI-powered video analytics tool that counts people leaving a metro station (or any crowded space) using computer vision. It uses **YOLOv8** for person detection and **ByteTrack** for tracking, then counts people crossing an exit zone and a direction-aware counting line. The result is an annotated video plus CSV / JSON analytics.

## Features

- Person detection and tracking (YOLOv8 + ByteTrack, centroid-tracker fallback)
- **Exit-zone counting**: each unique person is counted once when entering the zone
- **Line counting**: direction-aware IN / OUT / NET flow
- Track confirmation and crossing cooldown to avoid false counts
- Dwell-time estimation per person
- Crowd density levels (LOW / MEDIUM / HIGH) with a flashing **CROWD ALERT**
- Heat-map of foot positions (live overlay optional, PNG always saved)
- HUD with live crowd sparkline, FPS, peak crowd and average dwell time
- Draw the zone and line with your mouse (`--select-zone`, `--select-line`)
- Exports: annotated video, events CSV, per-second timeline CSV, summary JSON, heat-map PNG
- Optional ffmpeg H.264 compression and live preview window
- Fixes NumPy 2 / PyTorch incompatibility automatically

## Quick start

```bash
git clone https://github.com/<your-username>/metro-exit-counter.git
cd metro-exit-counter
pip install -r requirements.txt     # optional, main.py auto-installs missing packages
```

Place a video (`.mp4`, `.avi`, `.mov`, `.mkv`) in the folder, then:

```bash
python main.py
```

Results are written to the `outputs/` folder.

## Command-line options

| Option | Description |
|---|---|
| `--source PATH` | Video file or webcam index (default: auto-detect) |
| `--model NAME` | YOLO weights (default `yolov8n.pt`) |
| `--conf`, `--iou` | Detection confidence / NMS threshold |
| `--width N` | Processing width (default 960) |
| `--device` | `cpu` or GPU id such as `0` |
| `--skip N` | Detect every N+1 frames (faster) |
| `--max-frames N` | Stop after N frames |
| `--select-zone` | Draw the exit zone with the mouse |
| `--select-line` | Draw the counting line with the mouse |
| `--flip-line` | Swap IN / OUT direction |
| `--no-zone`, `--no-line` | Disable one of the counters |
| `--min-hits N`, `--cooldown N` | Anti-false-count tuning |
| `--density-medium N`, `--density-high N` | Crowd thresholds |
| `--heat-overlay` | Show heat-map inside the video |
| `--show` | Live preview (press `q` to quit) |
| `--compress` | Re-encode with ffmpeg if installed |
| `--no-export`, `--no-heatmap` | Skip CSV/JSON or heat-map files |

Examples:

```bash
python main.py --source metro.mp4 --show
python main.py --source metro.mp4 --select-zone --select-line --compress
python main.py --source 0 --show            # webcam
```

## Output files (`outputs/`)

| File | Content |
|---|---|
| `output_compressed.mp4` | Annotated video |
| `output_compressed_events.csv` | Every count event (frame, time, ID, type) |
| `output_compressed_timeline.csv` | Per-second crowd size and running totals |
| `output_compressed_summary.json` | Totals, peak crowd, average dwell time |
| `output_compressed_heatmap.png` | Where people walked most |

## Notes

- Camera angle matters: place the zone / line where people clearly pass.
- Use `--select-zone` and `--select-line` for best accuracy on your own footage.
- Requires Python 3.8+.
- Run the tests with `python tests/test_core.py`; build a container with `docker build -t metro-counter .`.

## Learn more

- [YOLO (object detection)](https://en.wikipedia.org/wiki/You_Only_Look_Once)
- [Object detection](https://en.wikipedia.org/wiki/Object_detection)
- [Computer vision](https://en.wikipedia.org/wiki/Computer_vision)
- [OpenCV](https://en.wikipedia.org/wiki/OpenCV)
- [Python](https://en.wikipedia.org/wiki/Python_(programming_language))
