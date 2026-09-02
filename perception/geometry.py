"""
geometry.py — общие функции проекции пикселей в 3D.

Используется и офлайн (demo_parser при разборе демонстраций),
и онлайн (postprocessor, aruco_tracker при инференсе),
чтобы процедура локализации была одинаковой в обоих контурах.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraIntrinsics:
    """Внутренние параметры камеры + масштаб карты глубины."""
    fx: float
    fy: float
    ppx: float
    ppy: float
    depth_scale: float = 0.001      # единица карты глубины -> метры

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraIntrinsics":
        """Читает intrinsics.json, сохранённый record_demo.py."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            fx=float(d["fx"]), fy=float(d["fy"]),
            ppx=float(d["ppx"]), ppy=float(d["ppy"]),
            depth_scale=float(d.get("depth_scale", 0.001)),
        )

    @classmethod
    def from_rs(cls, intr, depth_scale: float) -> "CameraIntrinsics":
        """Из объекта intrinsics библиотеки pyrealsense2."""
        return cls(
            fx=float(intr.fx), fy=float(intr.fy),
            ppx=float(intr.ppx), ppy=float(intr.ppy),
            depth_scale=float(depth_scale),
        )


def median_depth_raw(depth_frame: np.ndarray, cx: int, cy: int,
                     radius: int = 5, min_valid: int = 5) -> float | None:
    """
    Медианная глубина (в исходных единицах карты глубины, не в метрах)
    по квадратному патчу вокруг пикселя (cx, cy).
    Нулевые пиксели карты глубины считаются невалидными.
    Возвращает None, если валидных пикселей меньше min_valid.
    """
    h, w = depth_frame.shape[:2]
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    patch = depth_frame[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size < min_valid:
        return None
    return float(np.median(valid))


def deproject(cx: float, cy: float, depth_raw: float,
              intr: CameraIntrinsics) -> np.ndarray:
    """
    Обратная проекция пикселя (cx, cy) с глубиной depth_raw
    в систему координат камеры. Результат в метрах: [X, Y, Z].
    """
    z = depth_raw * intr.depth_scale
    x = (cx - intr.ppx) * z / intr.fx
    y = (cy - intr.ppy) * z / intr.fy
    return np.array([x, y, z], dtype=float)


def pixel_to_3d(cx: int, cy: int, depth_frame: np.ndarray,
                intr: CameraIntrinsics,
                radius: int = 5, min_valid: int = 5) -> np.ndarray | None:
    """
    Медианная глубина по патчу + обратная проекция.
    Возвращает [X, Y, Z] в метрах или None, если глубина невалидна.
    """
    d_raw = median_depth_raw(depth_frame, cx, cy, radius, min_valid)
    if d_raw is None:
        return None
    return deproject(cx, cy, d_raw, intr)