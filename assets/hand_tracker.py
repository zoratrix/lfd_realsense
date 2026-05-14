"""
hand_tracker.py — извлечение 3D траектории руки из записанной демонстрации

MediaPipe >= 0.10 (новый Tasks API)

Выход: numpy массив формы (T, 3) — траектория [x, y, z] в метрах,
       в системе координат камеры RealSense.

Использование:
    tracker = HandTracker()
    traj = tracker.extract_trajectory("data/demos/demo_cube_blue_0")
    # traj.shape == (N, 3)
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    raise SystemExit("mediapipe не установлен: pip install mediapipe")

# ---------------------------------------------------------------------------
# Параметры камеры RealSense D4xx (640x480)
# ---------------------------------------------------------------------------
CAM_FX = 615.0
CAM_FY = 615.0
CAM_CX = 320.0
CAM_CY = 240.0


def pixel_to_3d(px: float, py: float, depth_img: np.ndarray,
                patch: int = 5) -> np.ndarray | None:
    """Пиксельные координаты → 3D точка в метрах через depth."""
    h, w = depth_img.shape
    x0 = max(0, int(px) - patch // 2)
    y0 = max(0, int(py) - patch // 2)
    x1 = min(w, x0 + patch)
    y1 = min(h, y0 + patch)

    patch_depth = depth_img[y0:y1, x0:x1].astype(np.float32)
    valid = patch_depth[patch_depth > 0]
    if len(valid) == 0:
        return None

    z = float(np.median(valid)) / 1000.0  # мм → метры
    x = (px - CAM_CX) * z / CAM_FX
    y = (py - CAM_CY) * z / CAM_FY
    return np.array([x, y, z], dtype=np.float32)


class HandTracker:
    """
    Извлекает 3D траекторию запястья из папки с демонстрацией.
    Использует MediaPipe Hands Tasks API (>= 0.10).

    Args:
        model_path  : путь к hand_landmarker.task (скачивается автоматически)
        skip_frames : брать каждый N-й кадр (1 = все)
    """

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    DEFAULT_MODEL = Path("hand_landmarker.task")

    def __init__(self, model_path: str | Path | None = None, skip_frames: int = 1):
        self.skip_frames = skip_frames
        model_path = Path(model_path) if model_path else self.DEFAULT_MODEL

        # Скачиваем модель если нет
        if not model_path.exists():
            self._download_model(model_path)

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,  # оффлайн, кадр за кадром
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

    def _download_model(self, path: Path):
        import urllib.request
        logger.info(f"Скачиваю модель MediaPipe: {self.MODEL_URL}")
        print(f"Скачиваю hand_landmarker.task (~25 МБ)...")
        urllib.request.urlretrieve(self.MODEL_URL, path)
        print(f"Сохранено: {path}")

    def extract_trajectory(self, demo_path: str | Path) -> np.ndarray:
        """
        Читает все кадры демо, возвращает 3D траекторию запястья.

        Returns:
            np.ndarray shape (T, 3) — T точек [x, y, z] в метрах.
            Пустой массив (0, 3) если рука не найдена.
        """
        demo_path   = Path(demo_path)
        rgb_files   = sorted((demo_path / "rgb").glob("*.png"))
        depth_files = sorted((demo_path / "depth").glob("*.png"))

        if not rgb_files:
            logger.warning(f"Нет кадров: {demo_path}")
            return np.zeros((0, 3), dtype=np.float32)

        points: dict[int, np.ndarray] = {}
        total = len(rgb_files)

        for i, (rgb_fp, depth_fp) in enumerate(zip(rgb_files, depth_files)):
            if i % self.skip_frames != 0:
                continue

            rgb_bgr = cv2.imread(str(rgb_fp))
            depth   = cv2.imread(str(depth_fp), cv2.IMREAD_UNCHANGED)
            if rgb_bgr is None or depth is None:
                continue

            # MediaPipe Tasks API принимает mp.Image
            rgb_rgb  = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_rgb)
            result   = self._detector.detect(mp_image)

            if not result.hand_landmarks:
                continue

            # landmark 0 = запястье (WRIST)
            lm   = result.hand_landmarks[0][0]
            h, w = rgb_bgr.shape[:2]
            px   = lm.x * w
            py   = lm.y * h

            pt3d = pixel_to_3d(px, py, depth)
            if pt3d is not None:
                points[i] = pt3d

        if not points:
            logger.warning(f"Рука не найдена ни в одном кадре: {demo_path}")
            return np.zeros((0, 3), dtype=np.float32)

        found_pct = 100 * len(points) / total
        logger.info(f"  Рука найдена в {len(points)}/{total} кадрах ({found_pct:.0f}%)")

        return self._interpolate(points, total)

    def _interpolate(self, points: dict[int, np.ndarray], total: int) -> np.ndarray:
        """Линейная интерполяция внутренних пропусков."""
        indices    = sorted(points.keys())
        first, last = indices[0], indices[-1]
        n          = last - first + 1
        traj       = np.zeros((n, 3), dtype=np.float32)

        known_local = [idx - first for idx in indices]
        for axis in range(3):
            known_vals    = [points[idx][axis] for idx in indices]
            traj[:, axis] = np.interp(np.arange(n), known_local, known_vals)

        return traj

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("demo_path", help="Путь к папке демо (содержит rgb/ и depth/)")
    ap.add_argument("--skip",   type=int, default=1)
    ap.add_argument("--plot",   action="store_true")
    args = ap.parse_args()

    with HandTracker(skip_frames=args.skip) as tracker:
        traj = tracker.extract_trajectory(args.demo_path)

    if traj.shape[0] == 0:
        print("Траектория не извлечена — рука не найдена в кадрах.")
    else:
        print(f"\nТраектория: {traj.shape[0]} точек")
        print(f"  Старт : {traj[0]}")
        print(f"  Конец : {traj[-1]}")
        print(f"  Длина : {float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))):.3f} м")

        if args.plot:
            try:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(10, 6))
                ax  = fig.add_subplot(111, projection="3d")
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "b-", linewidth=1.5)
                ax.scatter(*traj[0],  color="green", s=80, label="старт")
                ax.scatter(*traj[-1], color="red",   s=80, label="конец")
                ax.set_xlabel("X (м)"); ax.set_ylabel("Y (м)"); ax.set_zlabel("Z (м)")
                ax.set_title(Path(args.demo_path).name)
                ax.legend()
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("pip install matplotlib")
