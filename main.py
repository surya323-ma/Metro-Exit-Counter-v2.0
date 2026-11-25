#!/usr/bin/env python3
"""
Metro Exit Counter  v2.0
========================
AI-powered video analytics for metro stations and other crowded places.

Features
  * YOLOv8 person detection + ByteTrack tracking (centroid-tracker fallback)
  * Exit-zone counting (polygon) AND direction-aware line counting (IN / OUT)
  * Track confirmation (ignores flickering false detections) + crossing cooldown
  * Dwell-time estimation per person
  * Crowd-density levels with on-screen CROWD ALERT
  * Live heat-map of foot positions (+ saved PNG)
  * HUD with live people-count sparkline
  * Interactive zone / line selection with the mouse
  * Exports: annotated video, events CSV, timeline CSV, summary JSON, heat-map PNG
  * Optional ffmpeg compression, live preview window, CLI configuration

Examples
  python main.py                              # auto-find a video in this folder
  python main.py --source metro.mp4 --show
  python main.py --source metro.mp4 --select-zone --select-line --compress
  python main.py --source 0                   # webcam
"""
import subprocess, sys, os


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0 – Fix NumPy / torch compatibility BEFORE any heavy imports
# ─────────────────────────────────────────────────────────────────────────────
def _pip(*args):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *args, "--quiet"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _pkg_version(name):
    try:
        import importlib.metadata as im
        return im.version(name)
    except Exception:
        return None


if os.environ.get("NUMPY_FIXED") != "1":
    np_ver = _pkg_version("numpy")
    if np_ver and int(np_ver.split(".")[0]) >= 2:
        print(f"[FIX] NumPy {np_ver} detected – downgrading to numpy<2 …")
        _pip("numpy<2", "--force-reinstall")
        print("[FIX] NumPy fixed. Restarting …\n")
        env = os.environ.copy()
        env["NUMPY_FIXED"] = "1"
        sys.exit(subprocess.run([sys.executable] + sys.argv, env=env).returncode)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 – Auto-install dependencies
# ─────────────────────────────────────────────────────────────────────────────
for import_name, pip_spec in {
    "ultralytics": "ultralytics",
    "cv2":         "opencv-python",
    "numpy":       "numpy<2",
}.items():
    try:
        __import__(import_name)
    except ImportError:
        print(f"[INSTALL] Installing {pip_spec} …")
        _pip(pip_spec)

# ─────────────────────────────────────────────────────────────────────────────
#  REAL IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import csv
import json
import shutil
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

warnings.filterwarnings("ignore")

__version__ = "2.0.0"
SCRIPT_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
#  DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
PERSON_CLASS = 0
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Relative (0-1) default exit zone and counting line
DEFAULT_ZONE = np.array([[0.04, 0.70], [0.96, 0.70],
                         [0.96, 0.98], [0.04, 0.98]], dtype=np.float32)
DEFAULT_LINE = np.array([[0.04, 0.60], [0.96, 0.60]], dtype=np.float32)

# BGR colours
COL_BOX_NORMAL = (30, 40, 210)
COL_BOX_ACTIVE = (0, 30, 255)
COL_TRAIL = (0, 200, 255)
COL_REGION = (0, 210, 255)
COL_HUD_TXT = (0, 255, 150)
COL_ACCENT = (0, 230, 255)
COL_COUNT = (30, 90, 255)
COL_ARROW = (0, 255, 200)
COL_LINE = (255, 200, 0)
COL_IN = (0, 220, 90)
COL_OUT = (60, 120, 255)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Metro Exit Counter – count people exiting through a zone / line.")
    p.add_argument("--source", default=None,
                   help="video file path or webcam index (default: auto-detect in folder)")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO weights (default yolov8n.pt)")
    p.add_argument("--out-dir", default="outputs", help="folder for all results")
    p.add_argument("--output", default="output_compressed.mp4", help="annotated video name")
    p.add_argument("--conf", type=float, default=0.40, help="detection confidence")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--width", type=int, default=960, help="processing width in pixels")
    p.add_argument("--device", default=None, help="'cpu', '0' (GPU) … (default: auto)")
    p.add_argument("--trail", type=int, default=28, help="trail length in frames")
    p.add_argument("--skip", type=int, default=0,
                   help="skip N frames between detections (speeds up, 0 = none)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")

    g = p.add_argument_group("counting")
    g.add_argument("--min-hits", type=int, default=5,
                   help="frames a track must exist before it can be counted")
    g.add_argument("--cooldown", type=int, default=20,
                   help="min frames between two line crossings of the same person")
    g.add_argument("--flip-line", action="store_true", help="swap IN / OUT direction")
    g.add_argument("--no-line", action="store_true", help="disable line counting")
    g.add_argument("--no-zone", action="store_true", help="disable exit-zone counting")
    g.add_argument("--select-zone", action="store_true", help="draw the exit zone with the mouse")
    g.add_argument("--select-line", action="store_true", help="draw the counting line with the mouse")

    g = p.add_argument_group("crowd")
    g.add_argument("--density-medium", type=int, default=4, help="people for MEDIUM density")
    g.add_argument("--density-high", type=int, default=8, help="people for HIGH density / alert")

    g = p.add_argument_group("output")
    g.add_argument("--show", action="store_true", help="live preview window (q = quit)")
    g.add_argument("--heat-overlay", action="store_true", help="show heat-map overlay in video")
    g.add_argument("--no-heatmap", action="store_true", help="skip heat-map PNG")
    g.add_argument("--compress", action="store_true", help="re-encode with ffmpeg (H.264) if installed")
    g.add_argument("--no-export", action="store_true", help="do not write CSV / JSON files")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def scale_pts(rel, fw, fh):
    p = np.array(rel, dtype=np.float32).copy()
    p[:, 0] *= fw
    p[:, 1] *= fh
    return p.astype(np.int32)


def point_in_poly(px, py, poly):
    n, inside, j = len(poly), False, len(poly) - 1
    x, y = float(px), float(py)
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def side_of_line(p, a, b):
    """Signed cross-product: >0 one side of the line, <0 the other."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def within_segment(p, a, b, margin=0.12):
    abx, aby = b[0] - a[0], b[1] - a[1]
    t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / max(abx * abx + aby * aby, 1e-6)
    return -margin <= t <= 1 + margin


# ─────────────────────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Track:
    tid: int
    first: int
    last: int
    trail: deque
    hits: int = 0
    in_zone: bool = False
    zone_counted: bool = False
    side: int = 0
    last_line_frame: int = -10 ** 9
    last_center: tuple = None


class CentroidTracker:
    """Very small fallback tracker, used only if ByteTrack is unavailable."""

    def __init__(self, max_dist=80, max_missing=15):
        self.next_id, self.objs, self.max_dist, self.max_missing = 1, {}, max_dist, max_missing

    def update(self, boxes):
        cents = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        assigned, used = [-1] * len(cents), set()
        for i, c in enumerate(cents):
            best, bd = None, self.max_dist
            for tid, (oc, _) in self.objs.items():
                if tid in used:
                    continue
                d = ((c[0] - oc[0]) ** 2 + (c[1] - oc[1]) ** 2) ** 0.5
                if d < bd:
                    best, bd = tid, d
            if best is None:
                best = self.next_id
                self.next_id += 1
            used.add(best)
            assigned[i] = best
            self.objs[best] = (c, 0)
        for tid in list(self.objs):
            if tid not in used:
                c, miss = self.objs[tid]
                if miss + 1 > self.max_missing:
                    del self.objs[tid]
                else:
                    self.objs[tid] = (c, miss + 1)
        return np.array(assigned, dtype=int)


class Heatmap:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.acc = np.zeros((h, w), np.float32)
        self._cache, self._cache_frame = None, -999

    def add(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.acc[y, x] += 1.0

    def colour(self):
        if self.acc.max() <= 0:
            return None, None
        blur = cv2.GaussianBlur(self.acc, (0, 0), sigmaX=max(8, self.w // 60))
        norm = (blur / max(blur.max(), 1e-6) * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET), norm

    def blend(self, frame, alpha=0.45, thresh=25):
        cmap, norm = self.colour()
        if cmap is None:
            return frame
        mask = (norm > thresh)[..., None]
        mixed = cv2.addWeighted(frame, 1 - alpha, cmap, alpha, 0)
        return np.where(mask, mixed, frame)

    def overlay_live(self, frame, fidx, every=15):
        if fidx - self._cache_frame >= every or self._cache is None:
            cmap, norm = self.colour()
            self._cache, self._cache_frame = (cmap, norm), fidx
        cmap, norm = self._cache
        if cmap is None:
            return frame
        mask = (norm > 25)[..., None]
        mixed = cv2.addWeighted(frame, 0.6, cmap, 0.4, 0)
        return np.where(mask, mixed, frame)


class Stats:
    """Collects events, per-second timeline and builds the summary."""

    def __init__(self, fps):
        self.fps = fps
        self.events = []        # (frame, time_s, track_id, event)
        self.timeline = []      # (time_s, tracked, zone_total, line_in, line_out)
        self.peak = 0
        self.peak_time = 0.0
        self.dwell_done = []
        self._last_sec = -1
        self.zone_count = 0
        self.line_in = 0
        self.line_out = 0

    def event(self, frame, tid, kind):
        self.events.append((frame, round(frame / self.fps, 2), int(tid), kind))

    def tick(self, frame, tracked):
        t = frame / self.fps
        if tracked > self.peak:
            self.peak, self.peak_time = tracked, t
        sec = int(t)
        if sec != self._last_sec:
            self._last_sec = sec
            self.timeline.append((sec, tracked, self.zone_count, self.line_in, self.line_out))

    def finish_track(self, tr: Track):
        if tr.hits >= 5:
            self.dwell_done.append((tr.last - tr.first) / self.fps)

    def avg_dwell(self, active):
        vals = list(self.dwell_done) + [(t.last - t.first) / self.fps for t in active if t.hits >= 5]
        return float(np.mean(vals)) if vals else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────────────────────────────────────
def corner_box(img, x1, y1, x2, y2, color, t=2, arm=16):
    for (ax, ay), (bx, by) in [
        ((x1, y1), (x1 + arm, y1)), ((x1, y1), (x1, y1 + arm)),
        ((x2, y1), (x2 - arm, y1)), ((x2, y1), (x2, y1 + arm)),
        ((x1, y2), (x1 + arm, y2)), ((x1, y2), (x1, y2 - arm)),
        ((x2, y2), (x2 - arm, y2)), ((x2, y2), (x2, y2 - arm)),
    ]:
        cv2.line(img, (ax, ay), (bx, by), color, t, cv2.LINE_AA)


def pill(img, text, x, y, fg=(255, 255, 255), bg=(25, 25, 25), scale=0.45, thick=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + bl), bg, -1)
    cv2.putText(img, text, (x, y), font, scale, fg, thick, cv2.LINE_AA)


def draw_zone(img, poly):
    ov = img.copy()
    cv2.fillPoly(ov, [poly], (0, 150, 255))
    cv2.addWeighted(ov, 0.16, img, 0.84, 0, img)
    cv2.polylines(img, [poly], True, COL_REGION, 2, cv2.LINE_AA)
    cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())
    pill(img, "EXIT ZONE", cx - 42, cy + 6, fg=(20, 220, 255), bg=(15, 15, 15), scale=0.52)


def draw_line(img, a, b, flash_in=False, flash_out=False):
    col = COL_IN if flash_in else COL_OUT if flash_out else COL_LINE
    cv2.line(img, tuple(a), tuple(b), col, 2, cv2.LINE_AA)
    mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
    # direction arrow (towards the "IN" side) – normal of the line
    nx, ny = -(b[1] - a[1]), (b[0] - a[0])
    n = max((nx * nx + ny * ny) ** 0.5, 1e-6)
    cv2.arrowedLine(img, (mx, my), (int(mx + nx / n * 26), int(my + ny / n * 26)),
                    COL_LINE, 2, cv2.LINE_AA, tipLength=0.4)
    pill(img, "COUNT LINE", a[0] + 4, a[1] - 8, fg=COL_LINE, bg=(15, 15, 15), scale=0.42)


def draw_trails(img, trails):
    for pts in trails.values():
        lst = list(pts)
        if len(lst) < 2:
            continue
        for i in range(1, len(lst)):
            a = i / len(lst)
            c = (int(COL_TRAIL[0] * a), int(COL_TRAIL[1] * a), int(COL_TRAIL[2] * a))
            cv2.line(img, lst[i - 1], lst[i], c, max(1, int(a * 2)), cv2.LINE_AA)


def draw_arrow(img, cx, cy, pcx, pcy):
    dx, dy = cx - pcx, cy - pcy
    d = (dx * dx + dy * dy) ** 0.5
    if d < 4:
        return
    s = 18 / d
    cv2.arrowedLine(img, (cx, cy), (int(cx + dx * s), int(cy + dy * s)),
                    COL_ARROW, 1, cv2.LINE_AA, tipLength=0.45)


def density_level(n, med, high):
    if n < med:
        return "LOW", (0, 200, 80)
    if n < high:
        return "MEDIUM", (0, 200, 255)
    return "HIGH", (0, 50, 230)


def hud(img, st: Stats, fps, tracked, avg_dwell, history, fw, fh, med, high, use_zone, use_line):
    W, H = 300, 236
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (W, H), (8, 8, 8), -1)
    cv2.addWeighted(ov, 0.68, img, 0.32, 0, img)
    fn, fnd = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX

    cv2.rectangle(img, (8, 6), (130, 24), (0, 170, 55), -1)
    cv2.putText(img, "  AI  ACTIVE", (10, 20), fn, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "METRO  EXIT  COUNTER", (8, 44), fn, 0.47, COL_ACCENT, 1, cv2.LINE_AA)
    cv2.line(img, (8, 50), (W - 8, 50), (55, 55, 55), 1)

    y = 78
    if use_zone:
        cv2.putText(img, f"EXIT ZONE:  {st.zone_count:03d}", (8, y), fnd, 0.66, COL_COUNT, 2, cv2.LINE_AA)
        y += 24
    if use_line:
        cv2.putText(img, f"IN {st.line_in:03d}", (8, y), fn, 0.55, COL_IN, 1, cv2.LINE_AA)
        cv2.putText(img, f"OUT {st.line_out:03d}", (92, y), fn, 0.55, COL_OUT, 1, cv2.LINE_AA)
        net = st.line_in - st.line_out
        cv2.putText(img, f"NET {net:+d}", (190, y), fn, 0.55, COL_ACCENT, 1, cv2.LINE_AA)
        y += 22
    cv2.putText(img, f"TRACKED : {tracked:02d}   PEAK : {st.peak:02d}", (8, y), fn, 0.46, COL_HUD_TXT, 1, cv2.LINE_AA)
    y += 18
    cv2.putText(img, f"AVG DWELL : {avg_dwell:5.1f}s   FPS : {fps:4.1f}", (8, y), fn, 0.46, COL_HUD_TXT, 1, cv2.LINE_AA)

    # sparkline of crowd size
    gx, gy, gw, gh = 8, H - 56, W - 16, 44
    cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (30, 30, 30), -1)
    if len(history) > 2:
        mx = max(max(history), 4)
        pts = [(gx + int(i / (len(history) - 1) * gw), gy + gh - 2 - int(v / mx * (gh - 6)))
               for i, v in enumerate(history)]
        cv2.polylines(img, [np.array(pts, np.int32)], False, COL_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(img, "crowd", (gx + 3, gy + 11), fn, 0.33, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (W, H), COL_ACCENT, 1)
    dens, dcol = density_level(tracked, med, high)
    pill(img, f"DENSITY: {dens}", fw - 178, 22, fg=dcol, bg=(12, 12, 12), scale=0.46)


def crowd_alert(img, fidx, fw):
    if (fidx // 8) % 2 == 0:
        cv2.rectangle(img, (fw // 2 - 130, 8), (fw // 2 + 130, 38), (0, 0, 190), -1)
        cv2.putText(img, "! CROWD ALERT !", (fw // 2 - 105, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def pick_points(frame, title, need_min, need_max, color):
    """Click points on the first frame. Enter = confirm, r = reset, Esc = cancel."""
    pts, win = [], f"{title}  [click | Enter=OK | r=reset | Esc=default]"

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < need_max:
            pts.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(view, p, 5, color, -1)
            if i:
                cv2.line(view, pts[i - 1], p, color, 2)
        if len(pts) >= 3 and need_max > 2:
            cv2.line(view, pts[-1], pts[0], color, 1)
        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts) >= need_min:
            break
        if k == ord("r"):
            pts.clear()
        if k == 27:
            pts = []
            break
    cv2.destroyWindow(win)
    return np.array(pts, dtype=np.int32) if pts else None


# ─────────────────────────────────────────────────────────────────────────────
#  EXPORTS
# ─────────────────────────────────────────────────────────────────────────────
def export_all(args, out_dir, st: Stats, active, meta):
    stem = Path(args.output).stem
    if not args.no_export:
        with open(out_dir / f"{stem}_events.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time_s", "track_id", "event"])
            w.writerows(st.events)
        with open(out_dir / f"{stem}_timeline.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "people_tracked", "exit_zone_total", "line_in_total", "line_out_total"])
            w.writerows(st.timeline)
        summary = {
            **meta,
            "exit_zone_count": st.zone_count,
            "line_in": st.line_in,
            "line_out": st.line_out,
            "net_flow": st.line_in - st.line_out,
            "peak_crowd": st.peak,
            "peak_crowd_time_s": round(st.peak_time, 1),
            "avg_dwell_time_s": round(st.avg_dwell(active), 2),
        }
        with open(out_dir / f"{stem}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[DONE] Events   → {out_dir / (stem + '_events.csv')}")
        print(f"[DONE] Timeline → {out_dir / (stem + '_timeline.csv')}")
        print(f"[DONE] Summary  → {out_dir / (stem + '_summary.json')}")


def ffmpeg_compress(path: Path):
    ff = shutil.which("ffmpeg")
    if not ff:
        print("[WARN] ffmpeg not found – skipping compression.")
        return
    tmp = path.with_suffix(".h264.mp4")
    r = subprocess.run([ff, "-y", "-i", str(path), "-vcodec", "libx264", "-crf", "26",
                        "-preset", "fast", "-pix_fmt", "yuv420p", str(tmp)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode == 0 and tmp.exists():
        before, after = path.stat().st_size, tmp.stat().st_size
        tmp.replace(path)
        print(f"[INFO] Compressed {before / 1e6:.1f} MB → {after / 1e6:.1f} MB")
    else:
        print("[WARN] ffmpeg failed – keeping original file.")


# ─────────────────────────────────────────────────────────────────────────────
#  INPUT / MODEL DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
def find_input_video(out_dir_name, out_name):
    skip = {out_name.lower(), "output.mp4"}
    for folder in (Path("."), SCRIPT_DIR):
        for p in sorted(folder.iterdir()):
            if (p.is_file() and p.suffix.lower() in VIDEO_EXTS
                    and p.name.lower() not in skip
                    and not p.name.lower().startswith("output")
                    and out_dir_name not in p.parts):
                return str(p)
    return None


def load_model(name):
    for cand in (Path(name), SCRIPT_DIR / name):
        if cand.exists():
            name = str(cand)
            break
    try:
        return YOLO(name)
    except Exception as e:
        print(f"[WARN] {name} failed ({e}) – trying yolo11n.pt …")
        return YOLO("yolo11n.pt")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = args.source or find_input_video(out_dir.name, args.output)
    if source is None:
        print("[ERROR] No input video found. Put a video next to main.py or use --source.")
        sys.exit(1)
    if str(source).isdigit():
        source = int(source)

    print(f"[INFO] Metro Exit Counter v{__version__}")
    print(f"[INFO] Input  : {source}")
    video_out = out_dir / args.output
    print(f"[INFO] Output : {video_out}")

    print(f"[INFO] Loading {args.model} …")
    model = load_model(args.model)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] OpenCV cannot open: {source}")
        sys.exit(1)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    if not fps_src or fps_src != fps_src or fps_src < 1:
        fps_src = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 0
    pw = args.width
    ph = int(orig_h * (pw / orig_w))
    print(f"[INFO] {orig_w}x{orig_h} @ {fps_src:.1f} fps ({total} frames) → {pw}x{ph}")

    # ── Zone & line (default / interactive) ──────────────────────────────────
    zone_poly = scale_pts(DEFAULT_ZONE, pw, ph)
    line_pts = scale_pts(DEFAULT_LINE, pw, ph)

    if args.select_zone or args.select_line:
        ok, first = cap.read()
        if ok:
            first = cv2.resize(first, (pw, ph))
            try:
                if args.select_zone:
                    sel = pick_points(first, "Select EXIT ZONE (3+ points)", 3, 12, COL_REGION)
                    if sel is not None:
                        zone_poly = sel
                if args.select_line:
                    sel = pick_points(first, "Select COUNT LINE (2 points)", 2, 2, COL_LINE)
                    if sel is not None:
                        line_pts = sel
            except cv2.error:
                print("[WARN] No GUI available – using default zone / line.")
            if not isinstance(source, int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    use_zone, use_line = not args.no_zone, not args.no_line
    la, lb = tuple(int(v) for v in line_pts[0]), tuple(int(v) for v in line_pts[1])
    deadband = 2.0 * max(((lb[0] - la[0]) ** 2 + (lb[1] - la[1]) ** 2) ** 0.5, 1)

    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (pw, ph))

    # ── State ────────────────────────────────────────────────────────────────
    tracks = {}
    trails = defaultdict(lambda: deque(maxlen=args.trail))
    stats = Stats(fps_src)
    heat = Heatmap(pw, ph)
    history = deque(maxlen=150)
    fallback_tracker = None
    use_fallback = False
    fps_sm, t_prev, fidx = fps_src, time.time(), 0
    flash_until = {"in": -1, "out": -1}
    last_boxes = None
    first_frame = None
    t_start = time.time()

    def detect(frame):
        nonlocal use_fallback, fallback_tracker
        kw = dict(classes=[PERSON_CLASS], conf=args.conf, iou=args.iou, verbose=False)
        if args.device is not None:
            kw["device"] = args.device
        if not use_fallback:
            try:
                r = model.track(frame, persist=True, tracker="bytetrack.yaml", **kw)[0]
                b = r.boxes
                if b is None or len(b) == 0 or b.id is None:
                    return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
                return (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                        b.id.cpu().numpy().astype(int))
            except Exception as e:
                print(f"\n[WARN] tracker error ({e}) – switching to centroid tracker")
                use_fallback = True
                fallback_tracker = CentroidTracker()
        r = model.predict(frame, **kw)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            fallback_tracker.update([])
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
        xyxy = b.xyxy.cpu().numpy()
        return xyxy, b.conf.cpu().numpy(), fallback_tracker.update(xyxy)

    print("[INFO] Processing … (Ctrl-C to stop early and still save results)\n")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fidx += 1
            if args.max_frames and fidx > args.max_frames:
                break
            frame = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_LINEAR)
            if first_frame is None:
                first_frame = frame.copy()

            # optional frame skipping: reuse last detections
            if args.skip and fidx % (args.skip + 1) != 1 and last_boxes is not None:
                xyxy, confs, ids = last_boxes
            else:
                xyxy, confs, ids = detect(frame)
                last_boxes = (xyxy, confs, ids)

            n_tracked = len(ids)
            seen = set()

            for (x1, y1, x2, y2), tid, cf in zip(xyxy, ids, confs):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                foot = (cx, y2)
                tid = int(tid)
                seen.add(tid)

                tr = tracks.get(tid)
                if tr is None:
                    tr = Track(tid, fidx, fidx, trails[tid])
                    tracks[tid] = tr
                tr.last = fidx
                tr.hits += 1
                tr.trail.append((cx, cy))
                heat.add(*foot)
                confirmed = tr.hits >= args.min_hits

                # ── exit-zone counting ──
                in_now = point_in_poly(foot[0], foot[1], zone_poly)
                if use_zone and in_now and not tr.in_zone and not tr.zone_counted and confirmed:
                    tr.zone_counted = True
                    stats.zone_count += 1
                    stats.event(fidx, tid, "EXIT_ZONE")
                tr.in_zone = in_now

                # ── line counting (direction aware) ──
                if use_line:
                    v = side_of_line(foot, la, lb)
                    side = 1 if v > deadband else -1 if v < -deadband else 0
                    if side != 0:
                        if (tr.side != 0 and side != tr.side and confirmed
                                and within_segment(foot, la, lb)
                                and fidx - tr.last_line_frame >= args.cooldown):
                            going_in = (tr.side > 0 and side < 0)
                            if args.flip_line:
                                going_in = not going_in
                            tr.last_line_frame = fidx
                            if going_in:
                                stats.line_in += 1
                                flash_until["in"] = fidx + 10
                                stats.event(fidx, tid, "LINE_IN")
                            else:
                                stats.line_out += 1
                                flash_until["out"] = fidx + 10
                                stats.event(fidx, tid, "LINE_OUT")
                        tr.side = side

                # ── drawing ──
                col = COL_BOX_ACTIVE if in_now else COL_BOX_NORMAL
                corner_box(frame, x1, y1, x2, y2, col)
                pill(frame, f"ID:{tid}", x1, y1 - 7, bg=(30, 30, 30), scale=0.40)
                pill(frame, f"{cf:.2f}", x2 - 34, y1 - 7, fg=(190, 190, 190), bg=(30, 30, 30), scale=0.36)
                if tr.last_center is not None:
                    draw_arrow(frame, cx, cy, *tr.last_center)
                tr.last_center = (cx, cy)

            # retire tracks that vanished
            for tid in [t for t, tr in tracks.items() if fidx - tr.last > 30]:
                stats.finish_track(tracks.pop(tid))
                trails.pop(tid, None)

            history.append(n_tracked)
            stats.tick(fidx, n_tracked)

            if args.heat_overlay:
                frame = heat.overlay_live(frame, fidx)
            draw_trails(frame, trails)
            if use_zone:
                draw_zone(frame, zone_poly)
            if use_line:
                draw_line(frame, la, lb, fidx <= flash_until["in"], fidx <= flash_until["out"])
            hud(frame, stats, fps_sm, n_tracked, stats.avg_dwell(tracks.values()),
                history, pw, ph, args.density_medium, args.density_high, use_zone, use_line)
            if n_tracked >= args.density_high:
                crowd_alert(frame, fidx, pw)
            pill(frame, f"FRAME {fidx:05d}", pw - 150, ph - 10,
                 fg=(140, 140, 140), bg=(10, 10, 10), scale=0.38)
            writer.write(frame)

            if args.show:
                cv2.imshow("Metro Exit Counter  (q = quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[INFO] Stopped by user.")
                    break

            t_now = time.time()
            fps_sm = 0.85 * fps_sm + 0.15 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now
            if fidx % 25 == 0:
                pct = f"{fidx / total * 100:5.1f}%" if total else "  n/a "
                print(f"\r  frame {fidx:5d}/{total or '?'}  {pct}  "
                      f"zone={stats.zone_count:3d} in={stats.line_in:3d} out={stats.line_out:3d} "
                      f"fps={fps_sm:4.1f}   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted – saving partial results …")
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    # ── Wrap-up ──────────────────────────────────────────────────────────────
    for tr in list(tracks.values()):
        stats.finish_track(tr)
    if not args.no_heatmap and first_frame is not None:
        hm_path = out_dir / f"{Path(args.output).stem}_heatmap.png"
        cv2.imwrite(str(hm_path), heat.blend(first_frame, alpha=0.55))
        print(f"\n[DONE] Heat-map → {hm_path}")
    if args.compress:
        ffmpeg_compress(video_out)

    meta = {
        "version": __version__,
        "source": str(source),
        "model": args.model,
        "processed_frames": fidx,
        "duration_s": round(fidx / fps_src, 1),
        "processing_time_s": round(time.time() - t_start, 1),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    export_all(args, out_dir, stats, [], meta)
    print(f"[DONE] Video    → {video_out}")
    print(f"[DONE] Exit-zone count : {stats.zone_count}")
    print(f"[DONE] Line IN / OUT   : {stats.line_in} / {stats.line_out}  (net {stats.line_in - stats.line_out:+d})")
    print(f"[DONE] Peak crowd      : {stats.peak}  at {stats.peak_time:.1f}s")


if __name__ == "__main__":
    main()
