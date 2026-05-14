"""
verify_trajectory.py — визуальная проверка траектории руки

Воспроизводит демо как видео с наложенной точкой запястья от MediaPipe.
Так можно сразу увидеть совпадает ли детектированная точка с реальной рукой.

Использование:
    python verify_trajectory.py data/demos/demo_cube_blue_0
    python verify_trajectory.py data/demos/demo_cube_blue_0 --save  # сохранить mp4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def verify(demo_path: str | Path, save: bool = False):
    demo_path = Path(demo_path)
    rgb_files   = sorted((demo_path / "rgb").glob("*.png"))
    depth_files = sorted((demo_path / "depth").glob("*.png"))

    if not rgb_files:
        print(f"Нет кадров в {demo_path}/rgb/")
        return

    # Импортируем трекер
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from assets.hand_tracker import HandTracker, pixel_to_3d

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        raise SystemExit("pip install mediapipe")

    # Создаём детектор в IMAGE режиме (надёжнее для проверки)
    model_path = Path(__file__).parent / "hand_landmarker.task"
    if not model_path.exists():
        model_path = Path("hand_landmarker.task")
    if not model_path.exists():
        raise SystemExit("hand_landmarker.task не найден — запусти сначала hand_tracker.py")

    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(opts)

    total = len(rgb_files)
    writer = None

    print(f"\nВоспроизведение: {demo_path.name}  ({total} кадров)")
    print("SPACE — пауза/продолжить    Q — выход\n")

    paused = False
    i = 0

    while i < total:
        rgb_fp   = rgb_files[i]
        depth_fp = depth_files[i] if i < len(depth_files) else None

        frame = cv2.imread(str(rgb_fp))
        depth = cv2.imread(str(depth_fp), cv2.IMREAD_UNCHANGED) if depth_fp else None

        if frame is None:
            i += 1
            continue

        vis = frame.copy()

        # Детекция
        rgb_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_rgb)
        result   = detector.detect(mp_img)

        h, w = frame.shape[:2]

        if result.hand_landmarks:
            lms = result.hand_landmarks[0]

            # Рисуем все 21 точку серым
            for lm in lms:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(vis, (px, py), 3, (150, 150, 150), -1)

            # Запястье (landmark 0) — большой зелёный кружок
            wrist = lms[0]
            wx, wy = int(wrist.x * w), int(wrist.y * h)
            cv2.circle(vis, (wx, wy), 10, (0, 255, 0), -1)
            cv2.circle(vis, (wx, wy), 10, (0, 180, 0), 2)

            # 3D координаты рядом с точкой
            if depth is not None:
                pt3d = pixel_to_3d(wrist.x * w, wrist.y * h, depth)
                if pt3d is not None:
                    label = f"x={pt3d[0]:.2f} y={pt3d[1]:.2f} z={pt3d[2]:.2f}m"
                    cv2.putText(vis, label, (wx + 14, wy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            status = "HAND OK"
            color  = (0, 220, 0)
        else:
            status = "NO HAND"
            color  = (0, 0, 220)

        # HUD
        cv2.putText(vis, status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vis, f"{i+1}/{total}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if paused:
            cv2.putText(vis, "PAUSE", (w - 90, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        # Прогресс-бар
        bar_w = int(w * (i / total))
        cv2.rectangle(vis, (0, h - 6), (bar_w, h), (0, 200, 0), -1)

        cv2.imshow(f"verify: {demo_path.name}", vis)

        if save:
            if writer is None:
                out_path = demo_path / "verify_trajectory.mp4"
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30, (w, h)
                )
                print(f"Запись в {out_path}")
            writer.write(vis)

        key = cv2.waitKey(33) & 0xFF  # ~30 fps
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused

        if not paused:
            i += 1

    detector.close()
    if writer:
        writer.release()
        print(f"Видео сохранено: {demo_path}/verify_trajectory.mp4")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("demo_path", help="Путь к папке демо")
    ap.add_argument("--save", action="store_true",
                    help="Сохранить результат в verify_trajectory.mp4")
    args = ap.parse_args()
    verify(args.demo_path, save=args.save)
