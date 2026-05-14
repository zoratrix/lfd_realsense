"""
debug_detections.py — визуальная проверка YOLO детекций на кадрах демонстрации.

Режимы:
  --mode video   : сохранить mp4 с нарисованными bbox (по умолчанию)
  --mode show    : показывать кадры в окне (нажать пробел — следующий, q — выход)
  --mode sample  : сохранить N отдельных png кадров равномерно по демо

Примеры:
  python debug_detections.py --demo demo_cube_green
  python debug_detections.py --demo demo_cube_green --mode show
  python debug_detections.py --demo demo_cube_green --mode sample --n-samples 20
  python debug_detections.py --demo demo_cube_green --start-frame 30 --end-frame 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Цвета для классов
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "cup_silver":      (200, 200, 200),
    "cup_dark_grey":   (80,  80,  80),
    "cube_blue":       (255, 100, 0),
    "cube_red":        (0,   0,  255),
    "cube_green":      (0,  200,  0),
    "washer_green":    (0,  255, 128),
    "lid":             (0,  200, 255),
    "aruco_container": (180, 0,  255),
}
DEFAULT_COLOR = (128, 128, 128)

# ---------------------------------------------------------------------------
# ArUco
# ---------------------------------------------------------------------------

_ARUCO_DICT   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_ARUCO_PARAMS = cv2.aruco.DetectorParameters()
_ARUCO_DET    = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)


def draw_aruco(img: np.ndarray) -> np.ndarray:
    corners, ids, _ = _ARUCO_DET.detectMarkers(img)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        for corner, mid in zip(corners, ids.flatten()):
            cx = int(corner[0, :, 0].mean())
            cy = int(corner[0, :, 1].mean())
            cv2.putText(img, f"ID={mid}", (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 0, 255), 2)
    return img


# ---------------------------------------------------------------------------
# Отрисовка детекций
# ---------------------------------------------------------------------------

def draw_detections(img: np.ndarray, detections: list[dict],
                    frame_idx: int, total: int) -> np.ndarray:
    out = img.copy()

    for det in detections:
        cls  = det["class_name"]
        conf = det["confidence"]
        x, y, w, h = det["bbox"]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cx, cy = int(x + w / 2), int(y + h / 2)

        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)

        # bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # центр
        cv2.circle(out, (cx, cy), 4, color, -1)

        # метка
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(out, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2),
                      color, -1)
        cv2.putText(out, label, (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # ArUco
    out = draw_aruco(out)

    # счётчик кадра
    cv2.putText(out, f"frame {frame_idx}/{total}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # кол-во детекций
    cv2.putText(out, f"detections: {len(detections)}",
                (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    return out


# ---------------------------------------------------------------------------
# Основная логика
# ---------------------------------------------------------------------------

def run(args):
    # --- конфиг ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    demo_cfg = next(
        (d for d in cfg["demos"] if d["id"] == args.demo), None
    )
    if demo_cfg is None:
        logger.error(f"Демо '{args.demo}' не найдено в {args.config}")
        sys.exit(1)

    demo_path = Path(demo_cfg["path"])
    obj_class = demo_cfg["object_class"]
    rgb_files = sorted((demo_path / "rgb").glob("*.png"))
    n = len(rgb_files)
    logger.info(f"Демо: {args.demo}  |  ожидаемый объект: {obj_class}  |  кадров: {n}")

    if n == 0:
        logger.error("Нет PNG кадров в rgb/")
        sys.exit(1)

    # --- диапазон кадров ---
    start = args.start_frame or 0
    end   = args.end_frame   or n
    end   = min(end, n)
    frames_to_process = rgb_files[start:end]
    logger.info(f"Обрабатываем кадры {start}…{end} ({len(frames_to_process)} шт.)")

    # --- детектор ---
    with open(args.perception) as f:
        pcfg = yaml.safe_load(f)
    m = pcfg["model"]
    sys.path.insert(0, ".")
    from perception.detector import ObjectDetector
    detector = ObjectDetector(
        weights_path=m["weights"],
        conf_thresh=args.conf or m["conf_thresh"],
        iou_thresh=m["iou_thresh"],
        device=m["device"],
    )
    logger.info(f"Детектор загружен, conf_thresh={args.conf or m['conf_thresh']}")

    # --- режим sample: равномерно N кадров ---
    if args.mode == "sample":
        out_dir = Path(args.out_dir) / args.demo
        out_dir.mkdir(parents=True, exist_ok=True)
        indices = np.linspace(0, len(frames_to_process) - 1,
                              args.n_samples, dtype=int)
        for rank, idx in enumerate(indices):
            fp = frames_to_process[idx]
            img = cv2.imread(str(fp))
            dets = detector.predict(img)
            vis  = draw_detections(img, dets, start + idx, n)
            out_path = out_dir / f"sample_{rank:03d}_frame{start+idx:05d}.png"
            cv2.imwrite(str(out_path), vis)
            det_names = [d["class_name"] for d in dets]
            logger.info(f"  кадр {start+idx:5d}: {det_names}")
        logger.info(f"Сохранено {len(indices)} кадров → {out_dir}")
        return

    # --- режим video ---
    if args.mode == "video":
        out_dir  = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.demo}_detections.mp4"

        first = cv2.imread(str(frames_to_process[0]))
        h, w  = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))

        for i, fp in enumerate(frames_to_process):
            img  = cv2.imread(str(fp))
            dets = detector.predict(img)
            vis  = draw_detections(img, dets, start + i, n)
            writer.write(vis)
            if i % 50 == 0:
                det_names = [d["class_name"] for d in dets]
                logger.info(f"  кадр {start+i:5d}: {det_names}")

        writer.release()
        logger.info(f"Видео сохранено → {out_path}")
        return

    # --- режим show ---
    if args.mode == "show":
        logger.info("Управление: пробел — следующий кадр, q — выход, "
                    "s — сохранить текущий кадр")
        out_dir = Path(args.out_dir) / args.demo
        step = args.step

        for i, fp in enumerate(frames_to_process[::step]):
            actual_idx = start + i * step
            img  = cv2.imread(str(fp))
            dets = detector.predict(img)
            vis  = draw_detections(img, dets, actual_idx, n)

            det_names = [f"{d['class_name']}({d['confidence']:.2f})" for d in dets]
            logger.info(f"  кадр {actual_idx:5d}: {det_names or 'нет детекций'}")

            cv2.imshow(f"detections — {args.demo}", vis)
            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                out_dir.mkdir(parents=True, exist_ok=True)
                save_path = out_dir / f"frame_{actual_idx:05d}.png"
                cv2.imwrite(str(save_path), vis)
                logger.info(f"  сохранён → {save_path}")

        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Визуальная проверка YOLO детекций")

    ap.add_argument("--config",      default="config/demos.yaml",
                    help="путь к demos.yaml")
    ap.add_argument("--perception",  default="config/perception.yaml",
                    help="путь к perception.yaml")
    ap.add_argument("--demo",        required=True,
                    help="id демонстрации из demos.yaml")
    ap.add_argument("--mode",        choices=["video", "show", "sample"],
                    default="video",
                    help="режим вывода (default: video)")
    ap.add_argument("--out-dir",     default="debug_out",
                    help="папка для сохранения результатов")
    ap.add_argument("--start-frame", type=int, default=None,
                    help="начальный кадр (default: 0)")
    ap.add_argument("--end-frame",   type=int, default=None,
                    help="конечный кадр (default: последний)")
    ap.add_argument("--conf",        type=float, default=None,
                    help="переопределить conf_thresh детектора")
    ap.add_argument("--fps",         type=int, default=15,
                    help="fps выходного видео (default: 15)")
    ap.add_argument("--step",        type=int, default=1,
                    help="шаг кадров в режиме show (default: 1)")
    ap.add_argument("--n-samples",   type=int, default=16,
                    help="кол-во кадров в режиме sample (default: 16)")

    args = ap.parse_args()
    run(args)