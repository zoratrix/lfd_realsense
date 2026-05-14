"""
record_demo.py — запись демонстраций для LfD pipeline
Версия 3: трекинг руки через MediaPipe (без маркеров)

Управление:
  SPACE  — старт / стоп записи
  Q      — выход (если запись идёт — останавливает и сохраняет)

Превью (слева направо):
  RGB с отрисовкой: скелет руки (MediaPipe) + ArUco контейнеры
  Colorized depth

Что важно при записи:
  - Рука должна быть видна в кадре — следи за зелёным скелетом в превью
  - Начинай запись когда объект лежит на конвейере ДО того как тянешься к нему
  - Заканчивай после того как положил объект в контейнер и убрал руку
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    raise SystemExit("pyrealsense2 не установлен: pip install pyrealsense2")

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from assets.hand_tracker import HandTracker as _HandTracker
    _HAND_MODEL = "hand_landmarker.task"
except ImportError:
    raise SystemExit("mediapipe не установлен: pip install mediapipe")

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

CONTAINER_IDS   = {1, 2, 3}      # ArUco ID контейнеров в сцене
ARUCO_DICT_NAME = cv2.aruco.DICT_4X4_50
STREAM_W, STREAM_H = 640, 480
STREAM_FPS = 30
OUTPUT_ROOT = Path("data/demos")

_aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
_aruco_params = cv2.aruco.DetectorParameters()
_aruco_det    = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)

# ---------------------------------------------------------------------------
# Отрисовка
# ---------------------------------------------------------------------------

# Детектор для превью (видеорежим — быстрее)
_LIVE_DETECTOR = None

def _get_live_detector():
    global _LIVE_DETECTOR
    if _LIVE_DETECTOR is None:
        import urllib.request
        model_path = _HAND_MODEL
        if not __import__('pathlib').Path(model_path).exists():
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            print(f"Скачиваю модель MediaPipe (~25 МБ)...")
            urllib.request.urlretrieve(url, model_path)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        _LIVE_DETECTOR = mp_vision.HandLandmarker.create_from_options(opts)
    return _LIVE_DETECTOR


def draw_preview(frame_bgr: np.ndarray,
                 timestamp_ms: int) -> tuple[np.ndarray, bool]:
    """
    Рисует скелет руки (MediaPipe) + ArUco контейнеры.
    Returns: (кадр с отрисовкой, найдена_ли_рука)
    """
    vis = frame_bgr.copy()

    # --- MediaPipe (новый API) ---
    rgb_rgb  = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_rgb)
    result   = _get_live_detector().detect_for_video(mp_image, timestamp_ms)
    hand_found = bool(result.hand_landmarks)

    if hand_found:
        # Рисуем точки вручную (landmark 0 = запястье — большой кружок)
        h, w = vis.shape[:2]
        for lm in result.hand_landmarks[0]:
            px, py = int(lm.x * w), int(lm.y * h)
            cv2.circle(vis, (px, py), 3, (0, 220, 0), -1)
        # Запястье крупнее
        wrist = result.hand_landmarks[0][0]
        cv2.circle(vis, (int(wrist.x * w), int(wrist.y * h)), 7, (0, 255, 0), -1)

    # --- ArUco контейнеры ---
    corners, ids, _ = _aruco_det.detectMarkers(vis)
    if ids is not None:
        for corner, mid in zip(corners, ids.flatten()):
            if int(mid) not in CONTAINER_IDS:
                continue
            pts = corner.astype(np.int32)
            cx  = int(corner[0, :, 0].mean())
            cy  = int(corner[0, :, 1].mean())
            cv2.polylines(vis, [pts], isClosed=True, color=(255, 130, 0), thickness=2)
            cv2.putText(vis, f"BOX[{mid}]", (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 130, 0), 2)

    # --- Статус ---
    status_txt   = "HAND OK" if hand_found else "NO HAND"
    status_color = (0, 220, 0) if hand_found else (0, 0, 220)
    cv2.putText(vis, status_txt, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    return vis, hand_found


# ---------------------------------------------------------------------------
# Фоновый поток записи
# ---------------------------------------------------------------------------

class FrameWriter(threading.Thread):
    def __init__(self, rgb_dir: Path, depth_dir: Path):
        super().__init__(daemon=True)
        self.rgb_dir   = rgb_dir
        self.depth_dir = depth_dir
        self.queue: Queue = Queue(maxsize=120)
        self._stop    = threading.Event()
        self.written  = 0

    def enqueue(self, idx: int, rgb: np.ndarray, depth: np.ndarray):
        try:
            self.queue.put_nowait((idx, rgb, depth))
        except Exception:
            pass

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set() or not self.queue.empty():
            try:
                idx, rgb, depth = self.queue.get(timeout=0.1)
            except Empty:
                continue
            name = f"{idx:05d}.png"
            cv2.imwrite(str(self.rgb_dir / name), rgb)
            cv2.imwrite(str(self.depth_dir / name), depth)
            self.written += 1


# ---------------------------------------------------------------------------
# Основная функция записи
# ---------------------------------------------------------------------------

def record_demo(demo_name: str) -> Path:
    out_dir   = OUTPUT_ROOT / demo_name
    rgb_dir   = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    pipeline  = rs.pipeline()
    config    = rs.config()
    config.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, STREAM_FPS)
    config.enable_stream(rs.stream.depth, STREAM_W, STREAM_H, rs.format.z16,  STREAM_FPS)
    pipeline.start(config)

    align     = rs.align(rs.stream.color)
    colorizer = rs.colorizer()

    recording  = False
    frame_idx  = 0
    timestamps = []
    writer: FrameWriter | None = None

    print(f"\n{'='*55}")
    print(f"  Демо: {demo_name}")
    print(f"  SPACE — старт/стоп      Q — выход и сохранение")
    print(f"  Следи за зелёным скелетом руки в превью!")
    print(f"{'='*55}\n")

    try:
        while True:
            frames    = pipeline.wait_for_frames()
            aligned   = align.process(frames)
            color_rs  = aligned.get_color_frame()
            depth_rs  = aligned.get_depth_frame()
            if not color_rs or not depth_rs:
                continue

            color_img = np.asanyarray(color_rs.get_data())
            depth_img = np.asanyarray(depth_rs.get_data())
            depth_vis = np.asanyarray(colorizer.colorize(depth_rs).get_data())

            rgb_ts_ns   = color_rs.get_timestamp() * 1_000_000
            depth_ts_ns = depth_rs.get_timestamp() * 1_000_000
            sys_ts_ns   = time.time_ns()

            ts_ms = int(time.time() * 1000)
            vis, hand_found = draw_preview(color_img, ts_ms)

            # Статус записи
            if recording:
                cv2.putText(vis, f"● REC  {frame_idx:05d}", (STREAM_W - 170, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 220), 2)
                if not hand_found:
                    cv2.putText(vis, "! HAND LOST", (10, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            else:
                cv2.putText(vis, "READY", (STREAM_W - 90, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2)

            preview = np.hstack([vis, depth_vis])
            cv2.imshow("LfD Demo Recorder  |  SPACE=rec  Q=quit", preview)

            if recording:
                writer.enqueue(frame_idx, color_img.copy(), depth_img.copy())
                timestamps.append({
                    "frame_idx":    frame_idx,
                    "rgb_ts_ns":    int(rgb_ts_ns),
                    "depth_ts_ns":  int(depth_ts_ns),
                    "system_ts_ns": int(sys_ts_ns),
                })
                frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if not recording:
                    writer = FrameWriter(rgb_dir, depth_dir)
                    writer.start()
                    recording = True
                    print("  ▶ Запись начата")
                else:
                    recording = False
                    writer.stop()
                    writer.join()
                    print(f"  ■ Остановлено. Кадров: {frame_idx}")
                    break
            elif key == ord("q"):
                if recording:
                    recording = False
                    writer.stop()
                    writer.join()
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if frame_idx == 0:
        print("  Ничего не записано.")
        return out_dir

    ts_path = out_dir / "timestamps.csv"
    with open(ts_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_idx","rgb_ts_ns","depth_ts_ns","system_ts_ns"])
        w.writeheader()
        w.writerows(timestamps)

    _make_video(rgb_dir, out_dir / "rgb.mp4")
    print(f"\n  ✓ Сохранено: {out_dir}  ({frame_idx} кадров)")
    return out_dir


def _make_video(rgb_dir: Path, out_path: Path):
    files = sorted(rgb_dir.glob("*.png"))
    if not files:
        return
    h, w = cv2.imread(str(files[0])).shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), STREAM_FPS, (w, h))
    for f in files:
        vw.write(cv2.imread(str(f)))
    vw.release()
    print(f"  rgb.mp4 → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help='Имя демо, напр. "demo_cube_blue_0"')
    record_demo(ap.parse_args().name)
