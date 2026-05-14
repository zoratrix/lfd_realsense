"""
demo_parser.py — парсинг демонстраций для LfD pipeline

Реализует алгоритм распознавания действий оператора на основе
совместного анализа временной сегментации видеопотока и изменений
состояния сцены (Scene State Delta).

Алгоритм:
    1. find_motion_onset() — детектор активности на оптическом потоке (cv2.absdiff),
       без YOLO. Находит момент когда объект устойчиво выехал на конвейере.

    2. S_start = snapshot(onset_idx … onset_idx + alpha_frames)
       S_end   = snapshot(last final_frames кадров)

    3. delta = для каждого класса объекта вычислить смещение S_start → S_end.
       Объекты со смещением > порога — acted_objects.

    4. Интерпретация:
       - lid_count уменьшился               → COVER
       - acted объект у контейнера          → MOVE(nearest_container)
       - acted объект не у контейнера       → FORWARD (уехал по конвейеру)
       - ничего не произошло                → PASS

    Принципиальное отличие от v1: алгоритм не требует знать целевой объект
    заранее. class и action_sequence извлекаются совместно из видео.
    demos.yaml нужен только для путей и верификации.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------

@dataclass
class ActionStep:
    action: str
    target_container_aruco_id: int | None = None

    def __repr__(self) -> str:
        if self.action == "MOVE" and self.target_container_aruco_id is not None:
            return f"MOVE(container_{self.target_container_aruco_id})"
        return self.action


@dataclass
class ObjectAlpha:
    shape: str
    color: str
    raw_class: str
    confidence: float
    detection_count: int


@dataclass
class SceneState:
    """
    Состояние сцены в момент времени.
    object_positions    : {class_name: (cx_px, cy_px)} — медианные позиции
    container_positions : {aruco_id: (cx_px, cy_px)}   — позиции контейнеров
    object_counts       : {class_name: int}             — медианное кол-во за кадр
    """
    object_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    container_positions: dict[int, tuple[float, float]] = field(default_factory=dict)
    object_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class DemoRecord:
    demo_id: str
    demo_path: Path
    object_class: str
    alpha: ObjectAlpha | None
    action_sequence: list[ActionStep]
    scene_start: SceneState
    scene_end: SceneState
    frame_count: int
    timestamps_ok: bool
    onset_idx: int

    @property
    def is_valid(self) -> bool:
        return (self.alpha is not None
                and self.frame_count > 0
                and len(self.action_sequence) > 0)


# ---------------------------------------------------------------------------
# shape/color из имени класса
# ---------------------------------------------------------------------------

CLASS_TO_ALPHA: dict[str, tuple[str, str]] = {
    "cup_silver":      ("cup",    "silver"),
    "cup_dark_grey":   ("cup",    "dark_grey"),
    "cube_blue":       ("cube",   "blue"),
    "cube_red":        ("cube",   "red"),
    "cube_green":      ("cube",   "green"),
    "washer_green":    ("washer", "green"),
    "lid":             ("lid",    "none"),
    "aruco_container": ("marker", "none"),
}

def class_name_to_alpha(cls: str, conf: float, count: int) -> ObjectAlpha:
    shape, color = CLASS_TO_ALPHA.get(cls, ("unknown", "unknown"))
    return ObjectAlpha(shape=shape, color=color, raw_class=cls,
                       confidence=conf, detection_count=count)


# ---------------------------------------------------------------------------
# ArUco хелпер
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
# Детектор начала движения
# ---------------------------------------------------------------------------

def find_motion_onset(
    rgb_files: list[Path],
    motion_threshold_px: int = 500,
    min_sustained_frames: int = 8,
    pixel_diff_thresh: int = 25,
    skip_first_n: int = 5,
) -> int:
    """
    Находит индекс первого кадра устойчивого движения без YOLO.

    Использует попиксельную разницу между соседними кадрами (cv2.absdiff).
    Движение считается устойчивым если motion_px > motion_threshold_px
    на протяжении min_sustained_frames кадров подряд.

    Returns:
        Индекс кадра начала устойчивого движения (или 0 как fallback)
    """
    prev_gray = None
    sustained = 0
    onset_candidate = 0

    for i, fp in enumerate(rgb_files):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and i >= skip_first_n:
            diff = cv2.absdiff(gray, prev_gray)
            _, mask = cv2.threshold(diff, pixel_diff_thresh, 255, cv2.THRESH_BINARY)
            motion_px = int(np.count_nonzero(mask))

            if motion_px > motion_threshold_px:
                if sustained == 0:
                    onset_candidate = i
                sustained += 1
            else:
                sustained = 0

            if sustained >= min_sustained_frames:
                logger.debug(
                    f"  motion onset: кадр {onset_candidate} "
                    f"({sustained} кадров подряд, {motion_px}px изменилось)"
                )
                return onset_candidate

        prev_gray = gray

    logger.debug("  motion onset: устойчивое движение не найдено → fallback 0")
    return 0


# ---------------------------------------------------------------------------
# Извлечение состояния сцены
# ---------------------------------------------------------------------------

def extract_scene_state(
    frame_files: list[Path],
    detector,
    container_ids: set[int],
    step: int = 3,
) -> SceneState:
    """
    Извлекает состояние сцены из набора кадров.

    Собирает медианные позиции объектов и медианное количество объектов
    каждого класса за кадр. Количество важно для детекции COVER:
    при наличии нескольких крышек медианная позиция почти не меняется,
    но count уменьшается когда одна крышка ложится на объект.
    """
    obj_positions: dict[str, list[tuple[float, float]]] = {}
    con_positions: dict[int,   list[tuple[float, float]]] = {}
    per_frame_counts: dict[str, list[int]] = {}

    for fp in frame_files[::step]:
        img = cv2.imread(str(fp))
        if img is None:
            continue

        # ArUco
        for cid, pos in detect_aruco_centers(img).items():
            if cid in container_ids:
                con_positions.setdefault(cid, []).append(pos)

        if detector is None:
            continue
        try:
            frame_cls_count: dict[str, int] = {}
            for det in detector.predict(img):
                cls = det["class_name"]
                if cls == "aruco_container":
                    continue
                x, y, w, h = det["bbox"]
                obj_positions.setdefault(cls, []).append((x + w / 2, y + h / 2))
                frame_cls_count[cls] = frame_cls_count.get(cls, 0) + 1
            for cls, cnt in frame_cls_count.items():
                per_frame_counts.setdefault(cls, []).append(cnt)
        except Exception as e:
            logger.debug(f"  YOLO ошибка {fp.name}: {e}")

    def median_pos(pts: list[tuple[float, float]]) -> tuple[float, float]:
        arr = np.array(pts)
        return (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))

    return SceneState(
        object_positions={
            cls: median_pos(pts)
            for cls, pts in obj_positions.items()
            if len(pts) >= 2
        },
        container_positions={
            cid: median_pos(pts)
            for cid, pts in con_positions.items()
            if len(pts) >= 2
        },
        object_counts={
            cls: int(round(np.median(counts)))
            for cls, counts in per_frame_counts.items()
            if len(counts) >= 2
        },
    )


# ---------------------------------------------------------------------------
# Вычисление delta
# ---------------------------------------------------------------------------

def compute_delta(s_start: SceneState, s_end: SceneState) -> dict[str, float]:
    """
    Смещение каждого объекта из S_start в S_end.
    inf = объект исчез из финального состояния.
    """
    delta = {}
    for cls, pos_start in s_start.object_positions.items():
        pos_end = s_end.object_positions.get(cls)
        if pos_end is None:
            delta[cls] = float("inf")
            logger.debug(f"  delta: '{cls}' исчез из S_end → inf")
        else:
            d = float(np.linalg.norm(np.array(pos_end) - np.array(pos_start)))
            delta[cls] = d
            logger.debug(f"  delta: '{cls}' смещение {d:.0f}px")
    return delta


# ---------------------------------------------------------------------------
# Интерпретация delta → action_sequence + alpha
# ---------------------------------------------------------------------------

def interpret_delta(
    s_start: SceneState,
    s_end: SceneState,
    move_threshold_px: float = 80.0,
    max_container_dist_px: float = 150.0,
) -> tuple[list[ActionStep], ObjectAlpha | None]:
    """
    Определяет action_sequence и alpha без знания целевого объекта.

    COVER детектируется через уменьшение медианного кол-ва крышек,
    а не через смещение их позиции (надёжнее при нескольких крышках).

    FORWARD — объект сместился/исчез, но не оказался у контейнера:
    уехал дальше по конвейеру.
    """
    delta = compute_delta(s_start, s_end)

    # --- COVER через подсчёт крышек ---
    lid_count_start = s_start.object_counts.get("lid", 0)
    lid_count_end   = s_end.object_counts.get("lid", 0)
    lid_disappeared = lid_count_start > 0 and lid_count_end < lid_count_start
    logger.debug(
        f"  interpret: lid count {lid_count_start}→{lid_count_end} "
        f"{'→ COVER' if lid_disappeared else '→ нет COVER'}"
    )

    # Объекты со значимым смещением (lid обрабатываем отдельно)
    acted = {
        cls: d for cls, d in delta.items()
        if d > move_threshold_px and cls != "lid"
    }

    if not acted and not lid_disappeared:
        logger.debug("  interpret: ничего не произошло → PASS")
        alpha = _alpha_from_state(s_start, exclude={"lid", "aruco_container"})
        return [ActionStep("PASS")], alpha

    actions: list[ActionStep] = []
    alpha: ObjectAlpha | None = None

    # --- MOVE / FORWARD ---
    for cls, displacement in acted.items():
        if displacement == float("inf"):
            obj_pos = s_start.object_positions[cls]
            logger.debug(f"  interpret: '{cls}' исчез → ищем контейнер по S_start позиции")
        else:
            obj_pos = s_end.object_positions[cls]
            logger.debug(f"  interpret: '{cls}' сместился {displacement:.0f}px")

        target_id = _nearest_container(
            obj_pos, s_end.container_positions,
            max_dist_px=max_container_dist_px,
        )

        if target_id is not None:
            actions.append(ActionStep("MOVE", target_id))
            logger.debug(f"  interpret: '{cls}' → MOVE(container_{target_id})")
        else:
            actions.append(ActionStep("FORWARD"))
            logger.debug(f"  interpret: '{cls}' → FORWARD (не у контейнера)")

        if alpha is None:
            alpha = class_name_to_alpha(cls, 1.0, len(s_start.object_positions))

    # --- COVER после MOVE/FORWARD ---
    if lid_disappeared:
        actions.insert(0, ActionStep("COVER"))
        if alpha is None:
            lid_pos = s_end.object_positions.get(
                "lid", s_start.object_positions.get("lid")
            )
            if lid_pos is not None:
                alpha = _nearest_object_alpha(
                    lid_pos, s_start, exclude={"lid", "aruco_container"}
                )

    if not actions:
        actions.append(ActionStep("PASS"))
    if alpha is None:
        alpha = _alpha_from_state(s_start, exclude={"lid", "aruco_container"})

    return actions, alpha


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _nearest_container(
    obj_pos: tuple[float, float],
    containers: dict[int, tuple[float, float]],
    max_dist_px: float = 150.0,
) -> int | None:
    if not containers:
        return None
    obj = np.array(obj_pos)
    dists = {cid: float(np.linalg.norm(obj - np.array(pos)))
             for cid, pos in containers.items()}
    nearest_id   = min(dists, key=dists.__getitem__)
    nearest_dist = dists[nearest_id]
    logger.debug(f"  ближайший контейнер: {nearest_id} ({nearest_dist:.0f}px)")
    if nearest_dist > max_dist_px:
        logger.debug(f"  слишком далеко ({nearest_dist:.0f}px > {max_dist_px}px) → None")
        return None
    return nearest_id


def _nearest_object_alpha(
    ref_pos: tuple[float, float],
    state: SceneState,
    exclude: set[str],
) -> ObjectAlpha | None:
    candidates = {cls: pos for cls, pos in state.object_positions.items()
                  if cls not in exclude}
    if not candidates:
        return None
    ref = np.array(ref_pos)
    nearest_cls = min(candidates,
                      key=lambda c: np.linalg.norm(ref - np.array(candidates[c])))
    return class_name_to_alpha(nearest_cls, 1.0, len(state.object_positions))


def _alpha_from_state(state: SceneState, exclude: set[str]) -> ObjectAlpha | None:
    for cls in state.object_positions:
        if cls not in exclude:
            return class_name_to_alpha(cls, 0.8, len(state.object_positions))
    return None


# ---------------------------------------------------------------------------
# DemoParser
# ---------------------------------------------------------------------------

class DemoParser:
    def __init__(self, config_path: str | Path, detector=None):
        self.config_path = Path(config_path)
        self.detector    = detector

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        g = self.config["global"]
        self.container_ids: set[int]    = set(g["container_ids"])
        self.alpha_frames: int          = g["alpha_frames"]
        self.final_frames: int          = g.get("final_frames", self.alpha_frames)
        self.motion_threshold_px: int   = g.get("motion_threshold_px", 500)
        self.min_sustained_frames: int  = g.get("min_sustained_frames", 8)
        self.move_threshold_px: float   = g.get("move_threshold_px", 80.0)
        self.max_container_dist_px: float = g.get("max_container_dist_px", 150.0)

        logger.info(f"DemoParser: {len(self.config['demos'])} демонстраций, "
                    f"контейнеры={self.container_ids}")

    def parse_all(self) -> list[DemoRecord]:
        records = []
        for demo_cfg in self.config["demos"]:
            rec = self._parse_one(demo_cfg)
            records.append(rec)
            status    = "✓" if rec.is_valid else "✗"
            acts      = " → ".join(str(a) for a in rec.action_sequence)
            alpha_str = (f"({rec.alpha.shape},{rec.alpha.color})"
                         if rec.alpha else "None")
            logger.info(f"  [{status}] {rec.demo_id}: "
                        f"α={alpha_str}, actions=[{acts}]")
        return records

    def parse_one(self, demo_id: str) -> DemoRecord | None:
        for demo_cfg in self.config["demos"]:
            if demo_cfg["id"] == demo_id:
                return self._parse_one(demo_cfg)
        return None

    def _parse_one(self, demo_cfg: dict) -> DemoRecord:
        demo_id   = demo_cfg["id"]
        demo_path = Path(demo_cfg["path"])
        obj_class = demo_cfg["object_class"]
        logger.info(f"Парсинг: {demo_id}")

        frame_count, ts_ok = self._verify_structure(demo_path)
        rgb_files = sorted((demo_path / "rgb").glob("*.png"))
        n = len(rgb_files)

        onset_idx = find_motion_onset(
            rgb_files,
            motion_threshold_px=self.motion_threshold_px,
            min_sustained_frames=self.min_sustained_frames,
        )
        logger.debug(f"  onset_idx: {onset_idx} / {n}")

        start_end   = min(onset_idx + self.alpha_frames, n)
        start_files = rgb_files[onset_idx:start_end]
        end_files   = rgb_files[max(0, n - self.final_frames):]

        if len(start_files) == 0:
            logger.warning(f"  S_start пустой, fallback → первые кадры")
            start_files = rgb_files[:self.alpha_frames]

        s_start = extract_scene_state(start_files, self.detector, self.container_ids)
        s_end   = extract_scene_state(end_files,   self.detector, self.container_ids)

        logger.debug(f"  S_start objects: {list(s_start.object_positions.keys())}")
        logger.debug(f"  S_start lid_count: {s_start.object_counts.get('lid', 0)}")
        logger.debug(f"  S_end   objects: {list(s_end.object_positions.keys())}")
        logger.debug(f"  S_end   lid_count: {s_end.object_counts.get('lid', 0)}")
        logger.debug(f"  S_end   containers: {list(s_end.container_positions.keys())}")

        action_sequence, alpha = interpret_delta(
            s_start, s_end,
            move_threshold_px=self.move_threshold_px,
            max_container_dist_px=self.max_container_dist_px,
        )

        if alpha is not None and alpha.raw_class != obj_class:
            logger.warning(f"  верификация: извлечён '{alpha.raw_class}', "
                           f"в конфиге '{obj_class}'")

        return DemoRecord(
            demo_id=demo_id, demo_path=demo_path, object_class=obj_class,
            alpha=alpha, action_sequence=action_sequence,
            scene_start=s_start, scene_end=s_end,
            frame_count=frame_count, timestamps_ok=ts_ok,
            onset_idx=onset_idx,
        )

    def _verify_structure(self, demo_path: Path) -> tuple[int, bool]:
        if not demo_path.exists():
            return 0, False
        rgb_files = sorted((demo_path / "rgb").glob("*.png"))
        if not rgb_files:
            return 0, False
        ts_ok = self._check_timestamps(demo_path / "timestamps.csv", len(rgb_files))
        return len(rgb_files), ts_ok

    def _check_timestamps(self, ts_file: Path, expected: int) -> bool:
        if not ts_file.exists():
            return False
        try:
            with open(ts_file) as f:
                rows = list(csv.DictReader(f))
            if len(rows) != expected:
                return False
            key = "rgb_ts_ns" if "rgb_ts_ns" in rows[0] else "rgb_timestamp_ns"
            vals = [int(r[key]) for r in rows]
            return all(vals[i] < vals[i+1] for i in range(len(vals)-1))
        except Exception:
            return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default="config/demos.yaml")
    ap.add_argument("--demo",    default=None)
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--debug",   action="store_true")
    args = ap.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    detector = None
    if not args.no_yolo:
        try:
            import sys, yaml as _yaml
            sys.path.insert(0, ".")
            from perception.detector import ObjectDetector
            with open("config/perception.yaml") as f:
                pcfg = _yaml.safe_load(f)
            m = pcfg["model"]
            detector = ObjectDetector(
                weights_path=m["weights"],
                conf_thresh=m["conf_thresh"],
                iou_thresh=m["iou_thresh"],
                device=m["device"],
            )
            logger.info("ObjectDetector загружен")
        except Exception as e:
            logger.warning(f"Detector недоступен: {e}")
            raise SystemExit(1)

    parser  = DemoParser(args.config, detector=detector)
    records = ([parser.parse_one(args.demo)] if args.demo
               else parser.parse_all())

    print("\n" + "=" * 60)
    valid = 0
    for rec in records:
        if rec is None:
            continue
        status = "✓ OK" if rec.is_valid else "✗ ПРОБЛЕМА"
        print(f"\n[{status}] {rec.demo_id}")
        print(f"  object_class (конфиг) : {rec.object_class}")
        if rec.alpha:
            match = "✓" if rec.alpha.raw_class == rec.object_class else "✗ РАСХОЖДЕНИЕ"
            print(f"  alpha (извлечён)      : shape={rec.alpha.shape}, "
                  f"color={rec.alpha.color}  [{match}]")
        else:
            print("  alpha                 : не определён")
        print(f"  onset_idx             : {rec.onset_idx}")
        print(f"  S_start lid_count     : {rec.scene_start.object_counts.get('lid', 0)}")
        print(f"  S_end   lid_count     : {rec.scene_end.object_counts.get('lid', 0)}")
        print(f"  S_start objects       : {list(rec.scene_start.object_positions.keys())}")
        print(f"  S_end   objects       : {list(rec.scene_end.object_positions.keys())}")
        print(f"  S_end   containers    : {list(rec.scene_end.container_positions.keys())}")
        print(f"  action_seq            : {' → '.join(str(a) for a in rec.action_sequence)}")
        print(f"  кадров                : {rec.frame_count}, ts_ok={rec.timestamps_ok}")
        if rec.is_valid:
            valid += 1

    print(f"\nВалидных: {valid}/{len([r for r in records if r])}")