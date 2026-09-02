"""
policy_runner.py — инференс LfD pipeline (без DMP)

Pipeline для одного кадра:
    1. YOLO → детекции → alpha (shape, color) главного объекта
    2. PolicyTrainer.predict(shape, color) → action_sequence
    3. ArUco → 2D позиции контейнеров
    4. RealSense depth → 3D координаты (объект + контейнеры)
    5. Возвращает RunnerResult — готово для передачи в ROS2 / MoveIt

PolicyRunner — чистый класс без ROS2-зависимостей.
Для интеграции с ROS2 создайте отдельный node, который вызывает runner.step().

Использование (standalone):
    python scripts/policy_runner.py --policy policy/policy.pkl --live
    python scripts/policy_runner.py --policy policy/policy.pkl --image frame.png --depth depth.png
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------

@dataclass
class Point3D:
    x: float  # метры, вправо
    y: float  # метры, вниз
    z: float  # метры, вперёд (глубина)

    def __repr__(self) -> str:
        return f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})m"


@dataclass
class ContainerTarget:
    aruco_id: int
    pos_2d: tuple[float, float]      # пиксели (cx, cy)
    pos_3d: Point3D | None           # None если depth недоступен


@dataclass
class RunnerResult:
    """
    Результат инференса для одного кадра.
    Передаётся в ROS2 node / MoveIt executor.
    """
    # Что за объект
    shape: str
    color: str
    raw_class: str
    object_pos_2d: tuple[float, float]       # пиксели
    object_pos_3d: Point3D | None            # None если depth недоступен

    # Что делать
    action_sequence: list[str]
    confidence: str                          # "exact" | "color" | "shape" | "fallback"
    warning: str | None

    # Куда (контейнеры, нужные для MOVE)
    container_targets: list[ContainerTarget] = field(default_factory=list)

    # Технические
    all_detections: list[dict] = field(default_factory=list)
    frame_shape: tuple[int, int] = (0, 0)   # (H, W)

    @property
    def primary_action(self) -> str:
        """Первое действие в последовательности."""
        return self.action_sequence[0] if self.action_sequence else "PASS"

    @property
    def move_target(self) -> ContainerTarget | None:
        """Контейнер для MOVE, если он есть в action_sequence."""
        for action in self.action_sequence:
            if action.startswith("MOVE(container_"):
                try:
                    aruco_id = int(action.split("container_")[1].rstrip(")"))
                    for ct in self.container_targets:
                        if ct.aruco_id == aruco_id:
                            return ct
                except (ValueError, IndexError):
                    pass
        return None

    def __repr__(self) -> str:
        acts = " → ".join(self.action_sequence)
        return (
            f"RunnerResult("
            f"alpha=({self.shape},{self.color}), "
            f"actions=[{acts}], "
            f"confidence={self.confidence})"
        )


# ---------------------------------------------------------------------------
# ArUco (переиспользуем из demo_parser логику)
# ---------------------------------------------------------------------------

_ARUCO_DICT   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_ARUCO_PARAMS = cv2.aruco.DetectorParameters()
_ARUCO_DET    = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)


def detect_aruco_centers(frame_bgr: np.ndarray) -> dict[int, tuple[float, float]]:
    corners, ids, _ = _ARUCO_DET.detectMarkers(frame_bgr)
    result = {}
    if ids is not None:
        for corner, mid in zip(corners, ids.flatten()):
            result[int(mid)] = (
                float(corner[0, :, 0].mean()),
                float(corner[0, :, 1].mean()),
            )
    return result


# ---------------------------------------------------------------------------
# Конвертация 2D → 3D через depth
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 0.001   # RealSense: 1 единица = 1 мм → 0.001 м

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CameraIntrinsics":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        cam = cfg["camera"]
        return cls(
            fx=cam["fx"],
            fy=cam["fy"],
            cx=cam["cx"],
            cy=cam["cy"],
            depth_scale=cam.get("depth_scale", 0.001),
        )

    @classmethod
    def from_realsense(cls, pipeline) -> "CameraIntrinsics":
        """
        Берём intrinsics прямо с камеры.
        Кадры выравниваются на color-поток (rs.align(rs.stream.color)),
        поэтому параметры берутся у color-потока, а не у depth.
        """
        rs = __import__("pyrealsense2")
        profile = pipeline.get_active_profile()
        color_profile = profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        return cls(fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
                   depth_scale=depth_scale)


def pixel_to_3d(
    u: float, v: float,
    depth_frame: np.ndarray,
    intrinsics: CameraIntrinsics,
    patch_radius: int = 3,
) -> Point3D | None:
    """
    Конвертирует пиксель (u, v) + depth → Point3D в метрах.

    Берём медиану по патчу patch_radius×patch_radius вокруг точки,
    чтобы подавить шум depth-сенсора.

    Returns None если глубина недоступна (нули, NaN).
    """
    h, w = depth_frame.shape[:2]
    u_i, v_i = int(round(u)), int(round(v))

    u0 = max(0, u_i - patch_radius)
    u1 = min(w, u_i + patch_radius + 1)
    v0 = max(0, v_i - patch_radius)
    v1 = min(h, v_i + patch_radius + 1)

    patch = depth_frame[v0:v1, u0:u1].astype(float)
    valid = patch[patch > 0]

    if valid.size == 0:
        logger.debug(f"  pixel_to_3d ({u:.0f},{v:.0f}): нет валидных depth-значений")
        return None

    d_raw = float(np.median(valid))
    d_m   = d_raw * intrinsics.depth_scale

    x = (u - intrinsics.cx) * d_m / intrinsics.fx
    y = (v - intrinsics.cy) * d_m / intrinsics.fy
    z = d_m

    return Point3D(x=x, y=y, z=z)


# ---------------------------------------------------------------------------
# PolicyRunner
# ---------------------------------------------------------------------------

# Копия из demo_parser — чтобы не импортировать весь модуль
_CLASS_TO_ALPHA: dict[str, tuple[str, str]] = {
    "cup_silver":      ("cup",    "silver"),
    "cup_dark_grey":   ("cup",    "dark_grey"),
    "cube_blue":       ("cube",   "blue"),
    "cube_red":        ("cube",   "red"),
    "cube_green":      ("cube",   "green"),
    "washer_green":    ("washer", "green"),
    "lid":             ("lid",    "none"),
    "aruco_container": ("marker", "none"),
}

# Классы которые не являются рабочими объектами
_SKIP_CLASSES = {"aruco_container", "lid"}


class PolicyRunner:
    """
    Инференс LfD pipeline для одного кадра.

    Параметры:
        policy_path     — путь к policy.pkl (сохранён build_policy.py)
        detector        — ObjectDetector (YOLO), или None (только ArUco)
        intrinsics      — CameraIntrinsics для 3D, или None (только 2D)
        container_ids   — ArUco ID контейнеров (из demos.yaml)
        min_confidence  — минимальный порог уверенности YOLO для alpha
    """

    def __init__(
        self,
        policy_path: str | Path,
        detector=None,
        intrinsics: CameraIntrinsics | None = None,
        container_ids: set[int] | None = None,
        min_confidence: float = 0.4,
    ):
        self.detector       = detector
        self.intrinsics     = intrinsics
        self.container_ids  = container_ids or set()
        self.min_confidence = min_confidence

        # Загружаем policy
        policy_path = Path(policy_path)
        # Импорт до pickle.load — иначе PolicyEntry не найдётся при десериализации
        from scripts.policy_trainer import PolicyTrainer, PolicyEntry  # noqa: F401

        with open(policy_path, "rb") as f:
            payload = pickle.load(f)
        self.trainer = PolicyTrainer()
        self.trainer.entries  = payload["entries"]
        self.trainer._clf     = payload["clf"]
        self.trainer._shapes  = payload["shapes"]
        self.trainer._colors  = payload["colors"]
        self.trainer._labels  = payload["labels"]

        meta = payload.get("meta", {})
        logger.info(
            f"PolicyRunner: policy загружена из {policy_path.name}, "
            f"alpha записей: {len(self.trainer.entries)}, "
            f"собрана: {meta.get('built_at', 'неизвестно')}"
        )

    # -----------------------------------------------------------------------
    # Главный метод
    # -----------------------------------------------------------------------

    def step(
        self,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray | None = None,
    ) -> RunnerResult | None:
        """
        Полный инференс для одного кадра.

        Args:
            rgb_frame   — BGR кадр (OpenCV формат)
            depth_frame — uint16 depth в мм (RealSense), или None

        Returns:
            RunnerResult или None если объект не обнаружен
        """
        h, w = rgb_frame.shape[:2]

        # 1. YOLO → детекции
        detections = self._detect(rgb_frame)
        if not detections:
            logger.debug("  runner: объектов не обнаружено")
            return None

        # 2. Выбираем главный объект → alpha
        main_det = self._pick_main_object(detections)
        if main_det is None:
            logger.debug("  runner: нет подходящего объекта (все SKIP или низкий conf)")
            return None

        cls_name   = main_det["class_name"]
        shape, color = _CLASS_TO_ALPHA.get(cls_name, ("unknown", "unknown"))
        x, y, bw, bh = main_det["bbox"]
        obj_cx, obj_cy = x + bw / 2, y + bh / 2

        logger.info(f"  runner: объект '{cls_name}' → alpha=({shape},{color})")

        # 3. 3D позиция объекта
        obj_3d = None
        if depth_frame is not None and self.intrinsics is not None:
            obj_3d = pixel_to_3d(obj_cx, obj_cy, depth_frame, self.intrinsics)
            logger.debug(f"  runner: объект 3D = {obj_3d}")

        # 4. Классификатор → action_sequence
        inference = self.trainer.predict(shape, color)
        if inference.warning:
            logger.warning(f"  runner ⚠: {inference.warning}")
        logger.info(
            f"  runner: action_sequence = {' → '.join(inference.action_sequence)} "
            f"[{inference.confidence}]"
        )

        # 5. ArUco → контейнеры
        aruco_centers = detect_aruco_centers(rgb_frame)
        container_targets = self._build_container_targets(
            aruco_centers, depth_frame
        )
        logger.debug(
            f"  runner: контейнеры = "
            f"{[f'id={ct.aruco_id} {ct.pos_3d}' for ct in container_targets]}"
        )

        return RunnerResult(
            shape=shape,
            color=color,
            raw_class=cls_name,
            object_pos_2d=(obj_cx, obj_cy),
            object_pos_3d=obj_3d,
            action_sequence=inference.action_sequence,
            confidence=inference.confidence,
            warning=inference.warning,
            container_targets=container_targets,
            all_detections=detections,
            frame_shape=(h, w),
        )

    # -----------------------------------------------------------------------
    # Визуализация
    # -----------------------------------------------------------------------

    def visualize(
        self,
        frame: np.ndarray,
        result: RunnerResult | None,
    ) -> np.ndarray:
        """
        Рисует результат инференса на кадре.
        Возвращает новый кадр (оригинал не изменяется).
        """
        vis = frame.copy()

        if result is None:
            cv2.putText(vis, "No object detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return vis

        # Bbox объекта
        for det in result.all_detections:
            if det["class_name"] in _SKIP_CLASSES:
                continue
            x, y, bw, bh = [int(v) for v in det["bbox"]]
            color = (0, 255, 0) if det["class_name"] == result.raw_class else (180, 180, 180)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(
                vis, f"{det['class_name']} {det['confidence']:.2f}",
                (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
            )

        # Контейнеры
        for ct in result.container_targets:
            cx, cy = int(ct.pos_2d[0]), int(ct.pos_2d[1])
            cv2.circle(vis, (cx, cy), 8, (255, 100, 0), 2)
            label = f"id={ct.aruco_id}"
            if ct.pos_3d:
                label += f" z={ct.pos_3d.z:.2f}m"
            cv2.putText(vis, label, (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

        # MOVE стрелка: объект → целевой контейнер
        if result.move_target is not None:
            ox, oy = int(result.object_pos_2d[0]), int(result.object_pos_2d[1])
            tx, ty = int(result.move_target.pos_2d[0]), int(result.move_target.pos_2d[1])
            cv2.arrowedLine(vis, (ox, oy), (tx, ty), (0, 200, 255), 2, tipLength=0.2)

        # HUD — alpha + actions + confidence
        acts_str = " → ".join(result.action_sequence)
        conf_color = {
            "exact":    (0, 255, 0),
            "color":    (0, 200, 255),
            "shape":    (0, 165, 255),
            "fallback": (0, 0, 255),
        }.get(result.confidence, (200, 200, 200))

        lines = [
            f"alpha: ({result.shape}, {result.color})  [{result.confidence}]",
            f"action: {acts_str}",
        ]
        if result.object_pos_3d:
            p = result.object_pos_3d
            lines.append(f"obj 3D: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m")
        if result.warning:
            lines.append(f"WARN: {result.warning[:60]}")

        for i, line in enumerate(lines):
            y_pos = 30 + i * 26
            cv2.putText(vis, line, (12, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, conf_color, 2)

        return vis

    # -----------------------------------------------------------------------
    # Вспомогательные методы
    # -----------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> list[dict]:
        if self.detector is None:
            return []
        try:
            return self.detector.predict(frame)
        except Exception as e:
            logger.error(f"  YOLO ошибка: {e}")
            return []

    def _pick_main_object(self, detections: list[dict]) -> dict | None:
        """
        Выбирает главный объект из детекций.

        Правила:
          - исключаем SKIP_CLASSES
          - исключаем ниже порога confidence
          - берём с наибольшей площадью bbox (ближайший / наиболее крупный)
        """
        candidates = [
            d for d in detections
            if d["class_name"] not in _SKIP_CLASSES
            and d["confidence"] >= self.min_confidence
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d["bbox"][2] * d["bbox"][3])

    def _build_container_targets(
        self,
        aruco_centers: dict[int, tuple[float, float]],
        depth_frame: np.ndarray | None,
    ) -> list[ContainerTarget]:
        targets = []
        ids_to_show = (
            self.container_ids if self.container_ids
            else set(aruco_centers.keys())
        )
        for aruco_id, (cx, cy) in aruco_centers.items():
            if aruco_id not in ids_to_show:
                continue
            pos_3d = None
            if depth_frame is not None and self.intrinsics is not None:
                pos_3d = pixel_to_3d(cx, cy, depth_frame, self.intrinsics)
            targets.append(ContainerTarget(
                aruco_id=aruco_id,
                pos_2d=(cx, cy),
                pos_3d=pos_3d,
            ))
        return sorted(targets, key=lambda t: t.aruco_id)


# ---------------------------------------------------------------------------
# CLI — тест без ROS2
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    ap = argparse.ArgumentParser(description="Тест policy_runner без ROS2")
    ap.add_argument("--policy",  default="policy/policy.pkl")
    ap.add_argument("--config",  default="config/perception.yaml")
    ap.add_argument("--image",   default=None, help="BGR .png кадр для теста")
    ap.add_argument("--depth",   default=None, help="uint16 depth .png (опционально)")
    ap.add_argument("--live",    action="store_true", help="тест с RealSense")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--debug",   action="store_true")
    args = ap.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    sys.path.insert(0, ".")

    # Detector
    detector = None
    if not args.no_yolo:
        from perception.detector import ObjectDetector
        with open(args.config) as f:
            pcfg = yaml.safe_load(f)
        m = pcfg["model"]
        detector = ObjectDetector(
            weights_path=m["weights"],
            conf_thresh=m["conf_thresh"],
            iou_thresh=m["iou_thresh"],
            device=m["device"],
        )

    # Intrinsics (опционально)
    intrinsics = None
    try:
        intrinsics = CameraIntrinsics.from_yaml(args.config)
        logger.info(f"Intrinsics загружены: fx={intrinsics.fx} fy={intrinsics.fy}")
    except (KeyError, FileNotFoundError):
        logger.info("Intrinsics не найдены в config — 3D недоступен")

    # container_ids из demos.yaml
    try:
        with open("config/demos.yaml") as f:
            dcfg = yaml.safe_load(f)
        container_ids = set(dcfg["global"]["container_ids"])
    except Exception:
        container_ids = set()

    runner = PolicyRunner(
        policy_path=args.policy,
        detector=detector,
        intrinsics=intrinsics,
        container_ids=container_ids,
    )

    # ── Статический кадр ────────────────────────────────────────────
    if args.image:
        rgb = cv2.imread(args.image)
        if rgb is None:
            logger.error(f"Не удалось загрузить: {args.image}")
            raise SystemExit(1)

        depth = None
        if args.depth:
            depth = cv2.imread(args.depth, cv2.IMREAD_ANYDEPTH)

        result = runner.step(rgb, depth)
        print("\n── Результат ──────────────────────────────")
        if result:
            print(f"  alpha          : ({result.shape}, {result.color})")
            print(f"  raw_class      : {result.raw_class}")
            print(f"  action_sequence: {' → '.join(result.action_sequence)}")
            print(f"  confidence     : {result.confidence}")
            print(f"  object_pos_2d  : {result.object_pos_2d}")
            print(f"  object_pos_3d  : {result.object_pos_3d}")
            print(f"  containers     : {result.container_targets}")
            if result.warning:
                print(f"  ⚠ warning      : {result.warning}")
        else:
            print("  Объект не обнаружен")

        vis = runner.visualize(rgb, result)
        cv2.imshow("policy_runner", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ── Live RealSense ───────────────────────────────────────────────
    elif args.live:
        try:
            import pyrealsense2 as rs
        except ImportError:
            logger.error("pyrealsense2 не установлен")
            raise SystemExit(1)

        pipeline = rs.pipeline()
        config   = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
        pipeline.start(config)

        if intrinsics is None:
            intrinsics = CameraIntrinsics.from_realsense(pipeline)
            runner.intrinsics = intrinsics
            logger.info(f"Intrinsics с камеры: fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f}")

        align = rs.align(rs.stream.color)
        logger.info("Live режим. Нажмите 'q' для выхода, 'i' для одного инференса.")

        try:
            while True:
                frames  = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_f = aligned.get_color_frame()
                depth_f = aligned.get_depth_frame()
                if not color_f or not depth_f:
                    continue

                bgr   = np.asanyarray(color_f.get_data())
                depth = np.asanyarray(depth_f.get_data())   # uint16, мм

                result = runner.step(bgr, depth)
                vis    = runner.visualize(bgr, result)

                cv2.imshow("policy_runner [live]", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("i") and result:
                    print(f"\n[i] {result}")
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

    else:
        ap.print_help()
